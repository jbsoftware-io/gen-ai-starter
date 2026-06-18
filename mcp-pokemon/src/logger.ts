import winston from 'winston';
import { AsyncLocalStorage } from 'async_hooks';

// Define context type
interface LogContext {
  [key: string]: unknown;
}

// Thread-safe context storage using AsyncLocalStorage
const contextStorage = new AsyncLocalStorage<LogContext>();

/**
 * Get the current context or an empty object
 */
function getCurrentContext(): LogContext {
  return contextStorage.getStore() || {};
}

/**
 * Extended logger interface with context methods
 */
interface ContextAwareLogger extends winston.Logger {
  setContext(key: string, value: unknown): void;
  setContext(context: LogContext): void;
  clearContext(): void;
}

/**
 * Create a Winston logger instance with JSON formatting and context methods
 */
function createLogger(): ContextAwareLogger {
  const logLevel = process.env.LOG_LEVEL || 'info';

  const baseLogger = winston.createLogger({
    level: logLevel,
    format: winston.format.combine(
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
      winston.format.errors({ stack: true }),
      winston.format.printf((info: any) => {
        // Merge current context into the log entry
        const context = getCurrentContext();
        const logEntry: any = {
          timestamp: info.timestamp,
          level: info.level,
          message: info.message,
          ...context,
          ...(info.meta && typeof info.meta === 'object' ? info.meta : {}),
        };

        // Include stack trace if present (for error level)
        if (info.stack) {
          logEntry.stack = info.stack;
        }

        return JSON.stringify(logEntry);
      })
    ),
    defaultMeta: {},
    transports: [
      new winston.transports.Console({
        stderrLevels: ['error'],
      }),
    ],
  });

  // Extend logger with context methods
  const contextAwareLogger = baseLogger as ContextAwareLogger;

  contextAwareLogger.setContext = function (keyOrContext: string | LogContext, value?: unknown): void {
    const context = getCurrentContext();
    
    if (typeof keyOrContext === 'string') {
      // Key-value form: setContext('key', value)
      context[keyOrContext] = value;
    } else {
      // Object form: setContext({ key1: value1, key2: value2 })
      Object.assign(context, keyOrContext);
    }
    
    contextStorage.enterWith(context);
  };

  contextAwareLogger.clearContext = function (): void {
    contextStorage.enterWith({});
  };

  return contextAwareLogger;
}

// Global logger instance
export const logger = createLogger();
