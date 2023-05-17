// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//import {Env} from './env';

//type LogLevelType = Env['logLevel'];

/** @type {Required<LogLevelType>} */
let logLevelValue = 'warning';

/** @type {Env} */
export const env = {
  /** @type {Env.WebAssemblyFlags} */
  wasm: {},
  /** @type {Env.WebGLFlags} */
  webgl: {},
  /** @type {Env.WebGpuFlags} */
  webgpu: {},

  /**
   * @type {LogLevelType}
   */
  set logLevel(value) {
    if (value === undefined) {
      return;
    }
    if (typeof value !== 'string' || ['verbose', 'info', 'warning', 'error', 'fatal'].indexOf(value) === -1) {
      throw new Error(`Unsupported logging level: ${value}`);
    }
    logLevelValue = value;
  },

  /**
   * @type {Required<LogLevelType>}
   */
  get logLevel() {
    return logLevelValue;
  },
};

// set property 'logLevel' so that they can be correctly transferred to worker by `postMessage()`.
Object.defineProperty(env, 'logLevel', {enumerable: true});
