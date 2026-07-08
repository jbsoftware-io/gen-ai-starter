#!/usr/bin/env node
/**
 * MCP Server generated from OpenAPI spec for pokemon-mcp v1.0.0
 * Generated on: 2026-03-21T13:11:36.353Z
 */

// Load environment variables from .env file
import dotenv from 'dotenv';
dotenv.config();

import { logger } from './logger.js';

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    ListPromptsRequestSchema,
    CallToolRequestSchema,
    ListToolsRequestSchema,
    type Tool,
    type CallToolResult,
    type CallToolRequest,
    GetPromptRequestSchema
} from "@modelcontextprotocol/sdk/types.js";
import { setupStreamableHttpServer } from "./streamable-http.js";

import { z, ZodError } from 'zod';
import { jsonSchemaToZod } from 'json-schema-to-zod';
import axios, { type AxiosRequestConfig, type AxiosError } from 'axios';
import { encode as encodeToon } from '@toon-format/toon';

/**
 * Type definition for JSON objects
 */
type JsonObject = Record<string, any>;

/**
 * Summarizes a Pokemon API response to extract only relevant fields
 * Reduces token count by ~40-50% compared to raw JSON response
 */
function summarizePokemon(data: any): JsonObject {
    if (!data || typeof data !== 'object') {
        return data;
    }

    // Extract basic fields
    const summary: JsonObject = {
        id: data.id,
        name: data.name,
        height: data.height,
        weight: data.weight,
    };

    // Extract types (array: [{name, slot}])
    if (Array.isArray(data.types)) {
        summary.types = data.types.map((t: any) => ({
            name: t.type?.name || t.name || '',
            slot: t.slot || 0,
        }));
    }

    // Extract base stats (array: [{stat, base_stat}])
    if (Array.isArray(data.stats)) {
        summary.stats = data.stats.map((s: any) => ({
            stat: s.stat?.name || s.stat || '',
            base_stat: s.base_stat || 0,
        }));
    }

    // Extract abilities (array: [{name, is_hidden}])
    if (Array.isArray(data.abilities)) {
        summary.abilities = data.abilities.map((a: any) => ({
            name: a.ability?.name || a.name || '',
            is_hidden: a.is_hidden || false,
        }));
    }

    // Extract moves (limit to first 10, array: [{name}])
    if (Array.isArray(data.moves)) {
        // Sort by learn method preference: level-up > egg > machine > others
        const moveMethodOrder: Record<string, number> = {
            'level-up': 0,
            'egg': 1,
            'machine': 2,
            'tutor': 3,
        };

        const sortedMoves = [...data.moves].sort((a: any, b: any) => {
            const aMethod = a.version_group_details?.[0]?.move_learn_method?.name || 'other';
            const bMethod = b.version_group_details?.[0]?.move_learn_method?.name || 'other';
            return (moveMethodOrder[aMethod] ?? 999) - (moveMethodOrder[bMethod] ?? 999);
        });

        summary.moves = sortedMoves.slice(0, 10).map((m: any) => ({
            name: m.move?.name || m.name || '',
        }));
    }

    return summary;
}

/**
 * Summarizes a Pokemon Species API response
 */
function summarizePokemonSpecies(data: any): JsonObject {
    if (!data || typeof data !== 'object') {
        return data;
    }

    const summary: JsonObject = {
        id: data.id,
        name: data.name,
        color: data.color?.name || '',
        habitat: data.habitat?.name || '',
        generation: data.generation?.name || '',
    };

    if (data.evolution_chain?.url) {
        summary.evolution_chain_url = data.evolution_chain.url;
    }

    if (Array.isArray(data.flavor_text_entries) && data.flavor_text_entries.length > 0) {
        const flavorEntry = data.flavor_text_entries.find((e: any) => e.language?.name === 'en');
        if (flavorEntry) {
            summary.description = flavorEntry.flavor_text?.replace(/\n/g, ' ').trim() || '';
        }
    }

    return summary;
}

/**
 * Summarizes a Type API response
 */
function summarizeType(data: any): JsonObject {
    if (!data || typeof data !== 'object') {
        return data;
    }

    const summary: JsonObject = {
        id: data.id,
        name: data.name,
    };

    if (data.damage_relations) {
        summary.damage_relations = {
            double_damage_from: data.damage_relations.double_damage_from?.slice(0, 5).map((t: any) => t.name || '') || [],
            double_damage_to: data.damage_relations.double_damage_to?.slice(0, 5).map((t: any) => t.name || '') || [],
            half_damage_from: data.damage_relations.half_damage_from?.slice(0, 5).map((t: any) => t.name || '') || [],
            half_damage_to: data.damage_relations.half_damage_to?.slice(0, 5).map((t: any) => t.name || '') || [],
            no_damage_from: data.damage_relations.no_damage_from?.map((t: any) => t.name || '') || [],
            no_damage_to: data.damage_relations.no_damage_to?.map((t: any) => t.name || '') || [],
        };
    }

    return summary;
}

/**
 * Summarizes an Ability API response
 */
function summarizeAbility(data: any): JsonObject {
    if (!data || typeof data !== 'object') {
        return data;
    }

    const summary: JsonObject = {
        id: data.id,
        name: data.name,
        is_main_series: data.is_main_series || false,
    };

    if (Array.isArray(data.effect_entries) && data.effect_entries.length > 0) {
        const enEffect = data.effect_entries.find((e: any) => e.language?.name === 'en');
        if (enEffect) {
            summary.effect = enEffect.effect || '';
        }
    }

    if (Array.isArray(data.flavor_text_entries) && data.flavor_text_entries.length > 0) {
        const enFlavor = data.flavor_text_entries.find((e: any) => e.language?.name === 'en');
        if (enFlavor) {
            summary.flavor_text = enFlavor.flavor_text?.replace(/\n/g, ' ').trim() || '';
        }
    }

    if (Array.isArray(data.pokemon)) {
        summary.pokemon_with_ability = data.pokemon.slice(0, 10).map((p: any) => ({
            name: p.pokemon?.name || '',
            is_hidden: p.is_hidden || false,
        }));
    }

    return summary;
}

/**
 * Summarizes a Move API response
 */
function summarizeMove(data: any): JsonObject {
    if (!data || typeof data !== 'object') {
        return data;
    }

    const summary: JsonObject = {
        id: data.id,
        name: data.name,
        power: data.power,
        accuracy: data.accuracy,
        priority: data.priority || 0,
        type: data.type?.name || '',
        category: data.damage_class?.name || '',
        pp: data.pp || 0,
    };

    if (Array.isArray(data.effect_entries) && data.effect_entries.length > 0) {
        const enEffect = data.effect_entries.find((e: any) => e.language?.name === 'en');
        if (enEffect) {
            summary.effect = enEffect.effect || '';
            summary.effect_chance = enEffect.effect_chance || null;
        }
    }

    return summary;
}

/**
 * Interface for MCP Tool Definition
 */
interface McpToolDefinition {
    name: string;
    description: string;
    inputSchema: any;
    method: string;
    pathTemplate: string;
    executionParameters: { name: string, in: string }[];
    requestBodyContentType?: string;
    securityRequirements: any[];
}

/**
 * Server configuration
 */
export const SERVER_NAME = "pokemon-mcp";
export const SERVER_VERSION = "1.0.0";
// Base URL for the API, can be set via environment variable or determined from OpenAPI spec
export const API_BASE_URL = process.env.API_BASE_URL || "https://pokeapi.co/api/v2";
logger.info("API_BASE_URL configured", { apiBaseUrl: API_BASE_URL });

/**
 * MCP Server instance
 */
const server = new Server(
    { name: SERVER_NAME, version: SERVER_VERSION },
    { capabilities: { tools: {} } }
);

/**
 * Map of tool definitions by name
 */
const toolDefinitionMap: Map<string, McpToolDefinition> = new Map([

    ["getPokemon", {
        name: "getPokemon",
        description: `Get a specific Pokémon by ID or name. Returns summarized data in TOON format (Token-Oriented Object Notation) for token efficiency: id, name, height, weight, types, base stats, abilities, and moves.`,
        inputSchema: { "type": "object", "properties": { "idOrName": { "type": "string", "description": "The ID or name of the Pokémon (e.g., 25 or 'pikachu')" } }, "required": ["idOrName"] },
        method: "get",
        pathTemplate: "/pokemon/{idOrName}",
        executionParameters: [{ "name": "idOrName", "in": "path" }],
        requestBodyContentType: undefined,
        securityRequirements: []
    }],
    ["getPokemonSpecies", {
        name: "getPokemonSpecies",
        description: `Get Pokémon species information by ID or name. Returns summarized data in TOON format: id, name, color, habitat, generation, evolution chain URL, and English description.`,
        inputSchema: { "type": "object", "properties": { "idOrName": { "type": "string", "description": "The ID or name of the Pokémon species (e.g., 25 or 'pikachu')" } }, "required": ["idOrName"] },
        method: "get",
        pathTemplate: "/pokemon-species/{idOrName}",
        executionParameters: [{ "name": "idOrName", "in": "path" }],
        requestBodyContentType: undefined,
        securityRequirements: []
    }],
    ["getType", {
        name: "getType",
        description: `Get Pokémon type information by ID or name. Returns TOON format data: id, name, and damage relations (double damage from/to, half damage from/to, no damage from/to).`,
        inputSchema: { "type": "object", "properties": { "idOrName": { "type": "string", "description": "The ID or name of the type (e.g., 1 or 'normal')" } }, "required": ["idOrName"] },
        method: "get",
        pathTemplate: "/type/{idOrName}",
        executionParameters: [{ "name": "idOrName", "in": "path" }],
        requestBodyContentType: undefined,
        securityRequirements: []
    }],
    ["getAbility", {
        name: "getAbility",
        description: `Get ability information by ID or name. Returns TOON format data: id, name, is_main_series, effect, flavor text, and list of Pokémon with this ability.`,
        inputSchema: { "type": "object", "properties": { "idOrName": { "type": "string", "description": "The ID or name of the ability (e.g., 1 or 'static')" } }, "required": ["idOrName"] },
        method: "get",
        pathTemplate: "/ability/{idOrName}",
        executionParameters: [{ "name": "idOrName", "in": "path" }],
        requestBodyContentType: undefined,
        securityRequirements: []
    }],
    ["getMove", {
        name: "getMove",
        description: `Get move information by ID or name. Returns TOON format data: id, name, power, accuracy, priority, type, category, pp, and effect description.`,
        inputSchema: { "type": "object", "properties": { "idOrName": { "type": "string", "description": "The ID or name of the move (e.g., 1 or 'pound')" } }, "required": ["idOrName"] },
        method: "get",
        pathTemplate: "/move/{idOrName}",
        executionParameters: [{ "name": "idOrName", "in": "path" }],
        requestBodyContentType: undefined,
        securityRequirements: []
    }],
]);

const promptDefinitionMap = {
    "pokemon-species-comparison": {
        name: "pokemon-species-comparison",
        description: "Compare top-level and species-level data (type, stats, abilities, moves, habitat, color, evolution chain) between two Pokémon",
        // The spec requires a flat array of arguments instead of a JSON inputSchema
        arguments: [
            {
                name: "pokemon1",
                description: "The name of the first Pokémon to compare",
                required: true
            },
            {
                name: "pokemon2",
                description: "The name of the second Pokémon to compare",
                required: true
            }
        ]
    }
};

/**
 * Security schemes from the OpenAPI spec
 */
const securitySchemes = {};

/**
 * Setup tool handlers for a server instance
 */
function setupToolHandlers(server: Server) {
    logger.debug("Setting up tool handlers", { toolCount: toolDefinitionMap.size });
    Array.from(toolDefinitionMap.keys()).forEach(toolName => {
        logger.debug("Registering tool", { toolName });
    });

    server.setRequestHandler(ListToolsRequestSchema, async () => {
        const toolsForClient: Tool[] = Array.from(toolDefinitionMap.values()).map(def => ({
            name: def.name,
            description: def.description,
            inputSchema: def.inputSchema
        }));
        return { tools: toolsForClient };
    });

    server.setRequestHandler(GetPromptRequestSchema, async (request) => {
        const { name, arguments: args } = request.params;

        if (name !== "pokemon-species-comparison") {
            throw new Error(`Prompt not found: ${name}`);
        }

        // Safely extract parameters passed from the Inspector UI
        const pokemon1 = args?.pokemon1 ?? "unknown";
        const pokemon2 = args?.pokemon2 ?? "unknown";

        // Build the context string dynamically using your variables
        const promptText = `You are a Pokémon species expert. Use \`getPokemon\` for ${pokemon1} and ${pokemon2} to compare: type, stats, abilities, and moves. Use \`getPokemonSpecies\` for ${pokemon1} and ${pokemon2} to compare: habitat, color, and evolution chain. Generate a markdown table and note any advantages/disadvantages.`;

        return {
            description: "Compare top-level and species-level data (type, stats, abilities, moves, habitat, color, evolution chain) between two Pokémon",
            messages: [
                {
                    role: "user",
                    content: {
                        type: "text",
                        text: promptText
                    }
                }
            ]
        };
    });

    server.setRequestHandler(ListPromptsRequestSchema, async () => {
        return {
            prompts: Object.values(promptDefinitionMap)
        };
    });

    server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest): Promise<CallToolResult> => {
        const { name: toolName, arguments: toolArgs } = request.params;
        logger.setContext('toolName', toolName);
        const toolDefinition = toolDefinitionMap.get(toolName);
        if (!toolDefinition) {
            logger.error(`Unknown tool requested: ${toolName}`);
            return { content: [{ type: "text", text: `Error: Unknown tool requested: ${toolName}` }] };
        }
        return await executeApiTool(toolName, toolDefinition, toolArgs ?? {}, securitySchemes);
    });
}

/**
 * Type definition for cached OAuth tokens
 */
interface TokenCacheEntry {
    token: string;
    expiresAt: number;
}

/**
 * Declare global __oauthTokenCache property for TypeScript
 */
declare global {
    var __oauthTokenCache: Record<string, TokenCacheEntry> | undefined;
}

/**
 * Acquires an OAuth2 token using client credentials flow
 * 
 * @param schemeName Name of the security scheme
 * @param scheme OAuth2 security scheme
 * @returns Acquired token or null if unable to acquire
 */
async function acquireOAuth2Token(schemeName: string, scheme: any): Promise<string | null | undefined> {
    try {
        // Check if we have the necessary credentials
        const clientId = process.env[`OAUTH_CLIENT_ID_SCHEMENAME`];
        const clientSecret = process.env[`OAUTH_CLIENT_SECRET_SCHEMENAME`];
        const scopes = process.env[`OAUTH_SCOPES_SCHEMENAME`];

        if (!clientId || !clientSecret) {
            logger.warn(`Missing client credentials for OAuth2 scheme`, { schemeName });
            return null;
        }

        // Initialize token cache if needed
        if (typeof global.__oauthTokenCache === 'undefined') {
            global.__oauthTokenCache = {};
        }

        // Check if we have a cached token
        const cacheKey = `${schemeName}_${clientId}`;
        const cachedToken = global.__oauthTokenCache[cacheKey];
        const now = Date.now();

        if (cachedToken && cachedToken.expiresAt > now) {
            logger.debug(`Using cached OAuth2 token`, { schemeName, expiresInSeconds: Math.floor((cachedToken.expiresAt - now) / 1000) });
            return cachedToken.token;
        }

        // Determine token URL based on flow type
        let tokenUrl = '';
        if (scheme.flows?.clientCredentials?.tokenUrl) {
            tokenUrl = scheme.flows.clientCredentials.tokenUrl;
            logger.debug(`Using client credentials flow`, { schemeName });
        } else if (scheme.flows?.password?.tokenUrl) {
            tokenUrl = scheme.flows.password.tokenUrl;
            logger.debug(`Using password flow`, { schemeName });
        } else {
            logger.warn(`No supported OAuth2 flow found`, { schemeName });
            return null;
        }

        // Prepare the token request
        let formData = new URLSearchParams();
        formData.append('grant_type', 'client_credentials');

        // Add scopes if specified
        if (scopes) {
            formData.append('scope', scopes);
        }

        logger.debug(`Requesting OAuth2 token`, { tokenUrl, schemeName });

        // Make the token request
        const response = await axios({
            method: 'POST',
            url: tokenUrl,
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`
            },
            data: formData.toString()
        });

        // Process the response
        if (response.data?.access_token) {
            const token = response.data.access_token;
            const expiresIn = response.data.expires_in || 3600; // Default to 1 hour

            // Cache the token
            global.__oauthTokenCache[cacheKey] = {
                token,
                expiresAt: now + (expiresIn * 1000) - 60000 // Expire 1 minute early
            };

            logger.info(`Successfully acquired OAuth2 token`, { schemeName, expiresInSeconds: expiresIn });
            return token;
        } else {
            logger.error(`Failed to acquire OAuth2 token: No access_token in response`, { schemeName });
            return null;
        }
    } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        logger.error(`Error acquiring OAuth2 token`, { schemeName, error: errorMessage });
        return null;
    }
}


/**
 * Executes an API tool with the provided arguments
 * 
 * @param toolName Name of the tool to execute
 * @param definition Tool definition
 * @param toolArgs Arguments provided by the user
 * @param allSecuritySchemes Security schemes from the OpenAPI spec
 * @returns Call tool result
 */
async function executeApiTool(
    toolName: string,
    definition: McpToolDefinition,
    toolArgs: JsonObject,
    allSecuritySchemes: Record<string, any>
): Promise<CallToolResult> {
    try {
        // Validate arguments against the input schema
        let validatedArgs: JsonObject;
        try {
            const zodSchema = getZodSchemaFromJsonSchema(definition.inputSchema, toolName);
            const argsToParse = (typeof toolArgs === 'object' && toolArgs !== null) ? toolArgs : {};
            validatedArgs = zodSchema.parse(argsToParse);
        } catch (error: unknown) {
            if (error instanceof ZodError) {
                const validationErrorMessage = `Invalid arguments for tool '${toolName}': ${error.errors.map(e => `${e.path.join('.')} (${e.code}): ${e.message}`).join(', ')}`;
                return { content: [{ type: 'text', text: validationErrorMessage }] };
            } else {
                const errorMessage = error instanceof Error ? error.message : String(error);
                return { content: [{ type: 'text', text: `Internal error during validation setup: ${errorMessage}` }] };
            }
        }

        // Prepare URL, query parameters, headers, and request body
        let urlPath = definition.pathTemplate;
        const queryParams: Record<string, any> = {};
        const headers: Record<string, string> = { 'Accept': 'application/json' };
        let requestBodyData: any = undefined;

        // Apply parameters to the URL path, query, or headers
        definition.executionParameters.forEach((param) => {
            const value = validatedArgs[param.name];
            if (typeof value !== 'undefined' && value !== null) {
                if (param.in === 'path') {
                    urlPath = urlPath.replace(`{${param.name}}`, encodeURIComponent(String(value)));
                }
                else if (param.in === 'query') {
                    queryParams[param.name] = value;
                }
                else if (param.in === 'header') {
                    headers[param.name.toLowerCase()] = String(value);
                }
            }
        });

        // Ensure all path parameters are resolved
        if (urlPath.includes('{')) {
            throw new Error(`Failed to resolve path parameters: ${urlPath}`);
        }

        // Construct the full URL
        const requestUrl = API_BASE_URL ? `${API_BASE_URL}${urlPath}` : urlPath;

        // Handle request body if needed
        if (definition.requestBodyContentType && typeof validatedArgs['requestBody'] !== 'undefined') {
            requestBodyData = validatedArgs['requestBody'];
            headers['content-type'] = definition.requestBodyContentType;
        }


        // Apply security requirements if available
        // Security requirements use OR between array items and AND within each object
        const appliedSecurity = definition.securityRequirements?.find(req => {
            // Try each security requirement (combined with OR)
            return Object.entries(req).every(([schemeName, scopesArray]) => {
                const scheme = allSecuritySchemes[schemeName];
                if (!scheme) return false;

                // API Key security (header, query, cookie)
                if (scheme.type === 'apiKey') {
                    return !!process.env[`API_KEY_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                }

                // HTTP security (basic, bearer)
                if (scheme.type === 'http') {
                    if (scheme.scheme?.toLowerCase() === 'bearer') {
                        return !!process.env[`BEARER_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                    }
                    else if (scheme.scheme?.toLowerCase() === 'basic') {
                        return !!process.env[`BASIC_USERNAME_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`] &&
                            !!process.env[`BASIC_PASSWORD_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                    }
                }

                // OAuth2 security
                if (scheme.type === 'oauth2') {
                    // Check for pre-existing token
                    if (process.env[`OAUTH_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`]) {
                        return true;
                    }

                    // Check for client credentials for auto-acquisition
                    if (process.env[`OAUTH_CLIENT_ID_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`] &&
                        process.env[`OAUTH_CLIENT_SECRET_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`]) {
                        // Verify we have a supported flow
                        if (scheme.flows?.clientCredentials || scheme.flows?.password) {
                            return true;
                        }
                    }

                    return false;
                }

                // OpenID Connect
                if (scheme.type === 'openIdConnect') {
                    return !!process.env[`OPENID_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                }

                return false;
            });
        });

        // If we found matching security scheme(s), apply them
        if (appliedSecurity) {
            // Apply each security scheme from this requirement (combined with AND)
            for (const [schemeName, scopesArray] of Object.entries(appliedSecurity)) {
                const scheme = allSecuritySchemes[schemeName];

                // API Key security
                if (scheme?.type === 'apiKey') {
                    const apiKey = process.env[`API_KEY_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                    if (apiKey) {
                        if (scheme.in === 'header') {
                            headers[scheme.name.toLowerCase()] = apiKey;
                            logger.debug(`Applied API key to header`, { schemeName, headerName: scheme.name });
                        }
                        else if (scheme.in === 'query') {
                            queryParams[scheme.name] = apiKey;
                            logger.debug(`Applied API key to query parameter`, { schemeName, paramName: scheme.name });
                        }
                        else if (scheme.in === 'cookie') {
                            // Add the cookie, preserving other cookies if they exist
                            headers['cookie'] = `${scheme.name}=${apiKey}${headers['cookie'] ? `; ${headers['cookie']}` : ''}`;
                            logger.debug(`Applied API key to cookie`, { schemeName, cookieName: scheme.name });
                        }
                    }
                }
                // HTTP security (Bearer or Basic)
                else if (scheme?.type === 'http') {
                    if (scheme.scheme?.toLowerCase() === 'bearer') {
                        const token = process.env[`BEARER_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                        if (token) {
                            headers['authorization'] = `Bearer ${token}`;
                            logger.debug(`Applied Bearer token`, { schemeName });
                        }
                    }
                    else if (scheme.scheme?.toLowerCase() === 'basic') {
                        const username = process.env[`BASIC_USERNAME_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                        const password = process.env[`BASIC_PASSWORD_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                        if (username && password) {
                            headers['authorization'] = `Basic ${Buffer.from(`${username}:${password}`).toString('base64')}`;
                            logger.debug(`Applied Basic authentication`, { schemeName });
                        }
                    }
                }
                // OAuth2 security
                else if (scheme?.type === 'oauth2') {
                    // First try to use a pre-provided token
                    let token = process.env[`OAUTH_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];

                    // If no token but we have client credentials, try to acquire a token
                    if (!token && (scheme.flows?.clientCredentials || scheme.flows?.password)) {
                        logger.debug(`Attempting to acquire OAuth token`, { schemeName });
                        token = (await acquireOAuth2Token(schemeName, scheme)) ?? '';
                    }

                    // Apply token if available
                    if (token) {
                        headers['authorization'] = `Bearer ${token}`;
                        logger.debug(`Applied OAuth2 token`, { schemeName });

                        // List the scopes that were requested, if any
                        const scopes = scopesArray as string[];
                        if (scopes && scopes.length > 0) {
                            logger.debug(`Requested OpenID Connect scopes`, { scopes });
                        }
                    }
                }
                // OpenID Connect
                else if (scheme?.type === 'openIdConnect') {
                    const token = process.env[`OPENID_TOKEN_${schemeName.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase()}`];
                    if (token) {
                        headers['authorization'] = `Bearer ${token}`;
                        logger.debug(`Applied OpenID Connect token`, { schemeName });

                        // List the scopes that were requested, if any
                        const scopes = scopesArray as string[];
                        if (scopes && scopes.length > 0) {
                            logger.debug(`Requested OAuth2 scopes`, { scopes, schemeName });
                        }
                    }
                }
            }
        }
        // Log warning if security is required but not available
        else if (definition.securityRequirements?.length > 0) {
            // First generate a more readable representation of the security requirements
            const securityRequirementsString = definition.securityRequirements
                .map(req => {
                    const parts = Object.entries(req)
                        .map(([name, scopesArray]) => {
                            const scopes = scopesArray as string[];
                            if (scopes.length === 0) return name;
                            return `${name} (scopes: ${scopes.join(', ')})`;
                        })
                        .join(' AND ');
                    return `[${parts}]`;
                })
                .join(' OR ');

            logger.warn(`Tool requires security but no credentials found`, { toolName, securityRequirements: securityRequirementsString });
        }


        // Prepare the axios request configuration
        const config: AxiosRequestConfig = {
            method: definition.method.toUpperCase(),
            url: requestUrl,
            params: queryParams,
            headers: headers,
            ...(requestBodyData !== undefined && { data: requestBodyData }),
        };

        // Log request info (doesn't affect MCP output)
        logger.setContext('endpoint', config.url);
        logger.info(`Executing tool`, { method: config.method, url: config.url });

        // Execute the request
        const response = await axios(config);

        // Process and format the response
        let responseText = '';
        const contentType = response.headers['content-type']?.toString().toLowerCase() || '';

        // Handle JSON responses
        if (contentType.includes('application/json') && typeof response.data === 'object' && response.data !== null) {
            try {
                // Use TOON format for all Pokemon tools
                if (toolName === 'getPokemon') {
                    const summarized = summarizePokemon(response.data);
                    responseText = encodeToon(summarized);
                } else if (toolName === 'getPokemonSpecies') {
                    const summarized = summarizePokemonSpecies(response.data);
                    responseText = encodeToon(summarized);
                } else if (toolName === 'getType') {
                    const summarized = summarizeType(response.data);
                    responseText = encodeToon(summarized);
                } else if (toolName === 'getAbility') {
                    const summarized = summarizeAbility(response.data);
                    responseText = encodeToon(summarized);
                } else if (toolName === 'getMove') {
                    const summarized = summarizeMove(response.data);
                    responseText = encodeToon(summarized);
                } else {
                    // For other tools, use standard JSON formatting
                    responseText = JSON.stringify(response.data, null, 2);
                }
            } catch (e) {
                responseText = "[Stringify Error]";
            }
        }
        // Handle string responses
        else if (typeof response.data === 'string') {
            responseText = response.data;
        }
        // Handle other response types
        else if (response.data !== undefined && response.data !== null) {
            responseText = String(response.data);
        }
        // Handle empty responses
        else {
            responseText = `(Status: ${response.status} - No body content)`;
        }

        // Return formatted response
        return {
            content: [
                {
                    type: "text",
                    text: `API Response (Status: ${response.status}):\n${responseText}`
                }
            ],
        };

    } catch (error: unknown) {
        // Handle errors during execution
        let errorMessage: string;

        // Format Axios errors specially
        if (axios.isAxiosError(error)) {
            errorMessage = formatApiError(error);
        }
        // Handle standard errors
        else if (error instanceof Error) {
            errorMessage = error.message;
        }
        // Handle unexpected error types
        else {
            errorMessage = 'Unexpected error: ' + String(error);
        }

        // Log error
        logger.error(`Error during execution of tool`, { errorMessage });

        // Return error message to client
        return { content: [{ type: "text", text: errorMessage }] };
    }
}


/**
 * Main function to start the server
 */
async function main() {
    // Create a server factory function that creates new Server instances per connection
    const serverFactory = () => {
        const newServer = new Server(
            { name: SERVER_NAME, version: SERVER_VERSION },
            { capabilities: { prompts: {}, tools: {} } }
        );
        // Setup tool handlers on the new server instance
        setupToolHandlers(newServer);
        return newServer;
    };

    try {
        logger.info("Starting MCP server", { port: 3001, apiBaseUrl: API_BASE_URL });
        await setupStreamableHttpServer(serverFactory, 3001);
        logger.info("MCP server started successfully", { port: 3001 });
    } catch (error) {
        logger.error("Error setting up StreamableHTTP server", { error: error instanceof Error ? error.message : String(error) });
        process.exit(1);
    }
}

/**
 * Cleanup function for graceful shutdown
 */
async function cleanup() {
    logger.info("Shutting down MCP server");
    process.exit(0);
}

// Register signal handlers
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// Start the server
main().catch((error) => {
    logger.error("Fatal error in main execution", { error: error instanceof Error ? error.message : String(error), stack: error instanceof Error ? error.stack : undefined });
    process.exit(1);
});

/**
 * Formats API errors for better readability
 * 
 * @param error Axios error
 * @returns Formatted error message
 */
function formatApiError(error: AxiosError): string {
    let message = 'API request failed.';
    if (error.response) {
        message = `API Error: Status ${error.response.status} (${error.response.statusText || 'Status text not available'}). `;
        const responseData = error.response.data;
        const MAX_LEN = 200;
        if (typeof responseData === 'string') {
            message += `Response: ${responseData.substring(0, MAX_LEN)}${responseData.length > MAX_LEN ? '...' : ''}`;
        }
        else if (responseData) {
            try {
                const jsonString = JSON.stringify(responseData);
                message += `Response: ${jsonString.substring(0, MAX_LEN)}${jsonString.length > MAX_LEN ? '...' : ''}`;
            } catch {
                message += 'Response: [Could not serialize data]';
            }
        }
        else {
            message += 'No response body received.';
        }
    } else if (error.request) {
        message = 'API Network Error: No response received from server.';
        if (error.code) message += ` (Code: ${error.code})`;
    } else {
        message += `API Request Setup Error: ${error.message}`;
    }
    return message;
}

/**
 * Converts a JSON Schema to a Zod schema for runtime validation
 * 
 * @param jsonSchema JSON Schema
 * @param toolName Tool name for error reporting
 * @returns Zod schema
 */
function getZodSchemaFromJsonSchema(jsonSchema: any, toolName: string): z.ZodTypeAny {
    if (typeof jsonSchema !== 'object' || jsonSchema === null) {
        return z.object({}).passthrough();
    }
    try {
        const zodSchemaString = jsonSchemaToZod(jsonSchema);
        const zodSchema = eval(zodSchemaString);
        if (typeof zodSchema?.parse !== 'function') {
            throw new Error('Eval did not produce a valid Zod schema.');
        }
        return zodSchema as z.ZodTypeAny;
    } catch (err: any) {
        logger.error(`Failed to generate/evaluate Zod schema`, { toolName, error: err instanceof Error ? err.message : String(err) });
        return z.object({}).passthrough();
    }
}
