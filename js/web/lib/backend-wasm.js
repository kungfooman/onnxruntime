// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {
  //Backend,
  env, InferenceSession,
  //SessionHandler
} from 'onnxruntime-common';
import {cpus} from 'os';

import {initializeWebAssemblyInstance} from './wasm/proxy-wrapper.js';
import {OnnxruntimeWebAssemblySessionHandler} from './wasm/session-handler.js';

/**
 * This function initializes all flags for WebAssembly.
 *
 * Those flags are accessible from `ort.env.wasm`. Users are allow to set those flags before the first inference session
 * being created, to override default value.
 * @returns {void}
 */
export const initializeFlags = () => {
  if (typeof env.wasm.initTimeout !== 'number' || env.wasm.initTimeout < 0) {
    env.wasm.initTimeout = 0;
  }

  if (typeof env.wasm.simd !== 'boolean') {
    env.wasm.simd = true;
  }

  if (typeof env.wasm.proxy !== 'boolean') {
    env.wasm.proxy = false;
  }

  if (typeof env.wasm.numThreads !== 'number' || !Number.isInteger(env.wasm.numThreads) || env.wasm.numThreads <= 0) {
    const numCpuLogicalCores = typeof navigator === 'undefined' ? cpus().length : navigator.hardwareConcurrency;
    env.wasm.numThreads = Math.min(4, Math.ceil((numCpuLogicalCores || 1) / 2));
  }
};
/**
 * @implements {Backend}
 */
class OnnxruntimeWebAssemblyBackend {
  /**
   * @returns {Promise<void>}
   */
  async init() {
    // populate wasm flags
    initializeFlags();

    // init wasm
    await initializeWebAssemblyInstance();
  }
  //createSessionHandler(path: string, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  //createSessionHandler(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SessionHandler>;
  /**
   *
   * @param {string|Uint8Array} pathOrBuffer
   * @param {InferenceSession.SessionOptions} [options]
   * @returns {Promise<SessionHandler>}
   */
  async createSessionHandler(pathOrBuffer, options) {
    const handler = new OnnxruntimeWebAssemblySessionHandler();
    await handler.loadModel(pathOrBuffer, options);
    return Promise.resolve(handler);
  }
}

export const wasmBackend = new OnnxruntimeWebAssemblyBackend();
