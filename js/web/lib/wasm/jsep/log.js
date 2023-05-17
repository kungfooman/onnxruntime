// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { env } from 'onnxruntime-common';

import {logLevelStringToEnum} from '../wasm-common.js';

//type LogLevel = NonNullable<Env['logLevel']>;
//type MessageString = string;
//type MessageFunction = () => string;
//type Message = MessageString|MessageFunction;

const logLevelPrefix = ['V', 'I', 'W', 'E', 'F'];
/**
 *
 * @param {number} level
 * @param {string} message
 * @returns {void}
 */
const doLog = (level, message) => {
  // eslint-disable-next-line no-console
  console.log(`[${logLevelPrefix[level]},${new Date().toISOString()}]${message}`);
};
/** @type {LogLevel|undefined} */
let configLogLevel;
/** @type {boolean|undefined} */
let debug;
/**
 * @param {LogLevel} $configLogLevel
 * @param {boolean} $debug
 * @returns {void}
 */
export const configureLogger = ($configLogLevel, $debug) => {
  configLogLevel = $configLogLevel;
  debug = $debug;
};

/**
 * A simple logging utility to log messages to the console.
 */
/**
 * @param {LogLevel} logLevel
 * @param {Message} msg
 * @returns {void}
 */
export const LOG = (logLevel, msg) => {
  const messageLevel = logLevelStringToEnum(logLevel);
  const configLevel = logLevelStringToEnum(configLogLevel);
  if (messageLevel >= configLevel) {
    doLog(messageLevel, typeof msg === 'function' ? msg() : msg);
  }
};
/**
 * A simple logging utility to log messages to the console. Only logs when debug is enabled.
 * args is Parameters<typeof LOG>
 * @type {typeof LOG}
 */
export const LOG_DEBUG = (...args) => {
  if (debug) {
    LOG(...args);
  }
};
