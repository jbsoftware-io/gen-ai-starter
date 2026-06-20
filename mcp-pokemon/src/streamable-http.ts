
/**
 * StreamableHTTP server setup for HTTP-based MCP communication using Hono
 */
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { serve } from '@hono/node-server';
import { v4 as uuid } from 'uuid';
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { InitializeRequestSchema, JSONRPCError } from "@modelcontextprotocol/sdk/types.js";
import { toReqRes, toFetchResponse } from 'fetch-to-node';

// Import server configuration constants
import { SERVER_NAME, SERVER_VERSION } from './index.js';

// Import logger utilities
import { logger } from './logger.js';
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";

// Constants
const SESSION_ID_HEADER_NAME = "mcp-session-id";
const JSON_RPC = "2.0";

/**
 * StreamableHTTP MCP Server handler
 */
class MCPStreamableHttpServer {
  serverFactory: () => Server;
  // Store active transports by session ID
  private transports: Record<string, WebStandardStreamableHTTPServerTransport> = {};

  constructor(serverFactory: () => Server) {
    this.serverFactory = serverFactory;
  }
  
  /**
   * Handle GET requests (typically used for static files)
   */
  async handleGetRequest(c: any) {
    logger.clearContext();
    logger.error("GET request received - StreamableHTTP transport only supports POST");
    return c.text('Method Not Allowed', 405, {
      'Allow': 'POST'
    });
  }
  
  /**
   * Handle POST requests (all MCP communication)
   */
  async handlePostRequest(c: any) {
    const sessionId = c.req.header(SESSION_ID_HEADER_NAME) || undefined;
    
    if (sessionId) {
      logger.setContext('sessionId', sessionId);
    }
    
    // DIAGNOSTIC 1: Log incoming request details completely
    logger.info(`[MCP DEBUG] POST request received`, { 
      sessionId: sessionId || 'none',
      url: c.req.raw.url,
      headers: Object.fromEntries(c.req.raw.headers.entries()) 
    });

    try {
      const rawRequest = c.req.raw;

      // DIAGNOSTIC 2: Safely inspect the JSON payload without consuming the stream
      try {
        const inspectClone = rawRequest.clone();
        const payloadText = await inspectClone.text();
        logger.info(`[MCP DEBUG] Incoming JSON-RPC Payload:`, { payload: payloadText });
      } catch (e) {
        logger.warn(`[MCP DEBUG] Could not inspect incoming request body text`, { error: String(e) });
      }

      // Reuse existing transport if we have an active session ID
      if (sessionId && this.transports[sessionId]) {
        const transport = this.transports[sessionId];
        logger.info(`[MCP DEBUG] Routing to existing session: ${sessionId}`);

        try {
          const webResponse = await transport.handleRequest(rawRequest);
          logger.info(`[MCP DEBUG] Existing session successfully produced response`, { status: webResponse.status });
          return webResponse;
        } catch (transportErr) {
          logger.error(`[MCP CRITICAL] Existing transport execution failed!`, {
            error: transportErr instanceof Error ? transportErr.message : String(transportErr),
            stack: transportErr instanceof Error ? transportErr.stack : 'No stack trace'
          });
          throw transportErr;
        }
      }

      // New Connection Initialization Routing
      logger.info("[MCP DEBUG] Initializing fresh connection pipeline...");
      const server = this.serverFactory();
      
      const transport = new WebStandardStreamableHTTPServerTransport({
        sessionIdGenerator: () => uuid(),
      });

      // DIAGNOSTIC 3: Explicitly bind the transport error hook
      transport.onerror = (err) => {
        logger.error('[MCP TRANSPORT ASYNC ERROR]', { 
          error: err instanceof Error ? err.message : String(err),
          stack: err instanceof Error ? err.stack : 'No stack trace available'
        });
      };

      // DIAGNOSTIC 4: Wrap the Server Connection handshake
      logger.info("[MCP DEBUG] Connecting transport layer to MCP Server instance...");
      await server.connect(transport);
      logger.info("[MCP DEBUG] Server-to-transport handshake successful");

      // DIAGNOSTIC 5: Capture execution point of the actual request resolution
      logger.info("[MCP DEBUG] Handing over execution to transport.handleRequest()...");
      let webResponse: Response;
      try {
        webResponse = await transport.handleRequest(rawRequest);
      } catch (handleErr) {
        logger.error(`[MCP CRITICAL] transport.handleRequest() crashed!`, {
          error: handleErr instanceof Error ? handleErr.message : String(handleErr),
          stack: handleErr instanceof Error ? handleErr.stack : 'No stack trace'
        });
        throw handleErr;
      }

      logger.info(`[MCP DEBUG] HandleRequest succeeded. Status: ${webResponse.status}`);

      const newSessionId = transport.sessionId;
      if (newSessionId) {
        logger.info(`[MCP DEBUG] New session ID registered`, { newSessionId });
        this.transports[newSessionId] = transport;

        transport.onclose = () => {
          logger.info(`[MCP DEBUG] Transport Session explicitly closed`, { newSessionId });
          delete this.transports[newSessionId];
        };
      }

      return webResponse;

    } catch (error: any) {
      logger.clearContext();
      
      // DIAGNOSTIC 6: Catch absolute root context traces
      logger.error('Error handling MCP request - Top Level Failure', { 
        message: error?.message || String(error),
        stack: error?.stack || 'No runtime stack trace',
        rawErrorObj: JSON.stringify(error, Object.getOwnPropertyNames(error))
      });

      return c.json(
        this.createErrorResponse(`Internal server error: ${error?.message || String(error)}`),
        500
      );
    }
  }
  
  /**
   * Create a JSON-RPC error response
   */
  private createErrorResponse(message: string): JSONRPCError {
    return {
      jsonrpc: JSON_RPC,
      error: {
        code: -32000,
        message: message,
      },
      id: uuid(),
    };
  }
  
  /**
   * Check if the request is an initialize request
   */
  private isInitializeRequest(body: any): boolean {
    const isInitial = (data: any) => {
      const result = InitializeRequestSchema.safeParse(data);
      return result.success;
    };
    
    if (Array.isArray(body)) {
      return body.some(request => isInitial(request));
    }
    
    return isInitial(body);
  }
}

/**
 * Sets up a web server for the MCP server using StreamableHTTP transport
 * 
 * @param serverFactory A factory function that creates new Server instances
 * @param port The port to listen on (default: 3001)
 * @returns The Hono app instance
 */
export async function setupStreamableHttpServer(serverFactory: () => Server, port = 3001) {
  // Create Hono app
  const app = new Hono();
  
  // Enable CORS
  app.use('*', cors());
  
  // Create MCP handler
  const mcpHandler = new MCPStreamableHttpServer(serverFactory);
  
  // Add a simple health check endpoint
  app.get('/health', (c) => {
    return c.json({ status: 'OK', server: SERVER_NAME, version: SERVER_VERSION });
  });
  
  // Main MCP endpoint supporting both GET and POST
  app.get("/mcp", (c) => mcpHandler.handleGetRequest(c));
  app.post("/mcp", (c) => mcpHandler.handlePostRequest(c));
  
  // Static files for the web client (if any)
  app.get('/*', async (c) => {
    const filePath = c.req.path === '/' ? '/index.html' : c.req.path;
    try {
      // Use Node.js fs to serve static files
      const fs = await import('fs');
      const path = await import('path');
      const { fileURLToPath } = await import('url');
      
      const __dirname = path.dirname(fileURLToPath(import.meta.url));
      const publicPath = path.join(__dirname, '..', '..', 'public');
      const fullPath = path.join(publicPath, filePath);
      
      // Simple security check to prevent directory traversal
      if (!fullPath.startsWith(publicPath)) {
        return c.text('Forbidden', 403);
      }
      
      try {
        const stat = fs.statSync(fullPath);
        if (stat.isFile()) {
          const content = fs.readFileSync(fullPath);
          
          // Set content type based on file extension
          const ext = path.extname(fullPath).toLowerCase();
          let contentType = 'text/plain';
          
          switch (ext) {
            case '.html': contentType = 'text/html'; break;
            case '.css': contentType = 'text/css'; break;
            case '.js': contentType = 'text/javascript'; break;
            case '.json': contentType = 'application/json'; break;
            case '.png': contentType = 'image/png'; break;
            case '.jpg': contentType = 'image/jpeg'; break;
            case '.svg': contentType = 'image/svg+xml'; break;
          }
          
          return new Response(content, {
            headers: { 'Content-Type': contentType }
          });
        }
      } catch (err) {
        // File not found or other error
        logger.clearContext();
        return c.text('Not Found', 404);
      }
    } catch (err) {
        logger.clearContext();
        logger.error('Error serving static file', { error: err instanceof Error ? err.message : String(err) });
    }
    
    logger.clearContext();
    return c.text('Not Found', 404);
  });
  
  // Start the server
  serve({
    fetch: app.fetch,
    port
  }, (info) => {
    logger.info(`MCP StreamableHTTP Server running`, { port: info.port, url: `http://localhost:${info.port}` });
    logger.info(`MCP Endpoint: http://localhost:${info.port}/mcp`);
    logger.info(`Health Check: http://localhost:${info.port}/health`);
  });
  
  return app;
}
