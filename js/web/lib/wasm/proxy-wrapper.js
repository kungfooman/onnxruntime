// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {
  //Env,
  env, InferenceSession} from 'onnxruntime-common';

//import {OrtWasmMessage, SerializableModeldata, SerializableSessionMetadata, SerializableTensor} from './proxy-messages.js';
import * as core from './wasm-core-impl.js';
import {initializeWebAssembly} from './wasm-factory.js';
/** @returns {boolean} */
const isProxy = () => !!env.wasm.proxy && typeof document !== 'undefined';
/** @type {Worker|undefined} */
let proxyWorker;
let initializing = false;
let initialized = false;
let aborted = false;

// resolve; reject
//type PromiseCallbacks<T = void> = [(result: T) => void, (reason: unknown) => void];

/** @type {PromiseCallbacks} */
let initWasmCallbacks;
/** @type {PromiseCallbacks} */
let initOrtCallbacks;
/** @type {Array<PromiseCallbacks<SerializableModeldata>>} */
const createSessionAllocateCallbacks = [];
/** @type {Array<PromiseCallbacks<SerializableSessionMetadata>>} */
const createSessionFinalizeCallbacks = [];
/** @type {Array<PromiseCallbacks<SerializableSessionMetadata>>} */
const createSessionCallbacks = [];
/** @type {Array<PromiseCallbacks<void>>} */
const releaseSessionCallbacks = [];
/** @type {Array<PromiseCallbacks<SerializableTensor[]>>} */
const runCallbacks = [];
/** @type {Array<PromiseCallbacks<void>>} */
const endProfilingCallbacks = [];

/**
 * @throws {Error}
 * @returns {void}
 */
const ensureWorker = () => {
  if (initializing || !initialized || aborted || !proxyWorker) {
    throw new Error('worker not ready');
  }
};
/**
 * @param {MessageEvent<OrtWasmMessage>} ev
 * @returns {void}
 */
const onProxyWorkerMessage = (ev) => {
  switch (ev.data.type) {
    case 'init-wasm':
      initializing = false;
      if (ev.data.err) {
        aborted = true;
        initWasmCallbacks[1](ev.data.err);
      } else {
        initialized = true;
        initWasmCallbacks[0]();
      }
      break;
    case 'init-ort':
      if (ev.data.err) {
        initOrtCallbacks[1](ev.data.err);
      } else {
        initOrtCallbacks[0]();
      }
      break;
    case 'create_allocate':
      if (ev.data.err) {
        createSessionAllocateCallbacks.shift()/*!*/[1](ev.data.err);
      } else {
        createSessionAllocateCallbacks.shift()/*!*/[0](ev.data.out/*!*/);
      }
      break;
    case 'create_finalize':
      if (ev.data.err) {
        createSessionFinalizeCallbacks.shift()/*!*/[1](ev.data.err);
      } else {
        createSessionFinalizeCallbacks.shift()/*!*/[0](ev.data.out/*!*/);
      }
      break;
    case 'create':
      if (ev.data.err) {
        createSessionCallbacks.shift()/*!*/[1](ev.data.err);
      } else {
        createSessionCallbacks.shift()/*!*/[0](ev.data.out/*!*/);
      }
      break;
    case 'release':
      if (ev.data.err) {
        releaseSessionCallbacks.shift()/*!*/[1](ev.data.err);
      } else {
        releaseSessionCallbacks.shift()/*!*/[0]();
      }
      break;
    case 'run':
      if (ev.data.err) {
        runCallbacks.shift()/*!*/[1](ev.data.err);
      } else {
        runCallbacks.shift()/*!*/[0](ev.data.out/*!*/);
      }
      break;
    case 'end-profiling':
      if (ev.data.err) {
        endProfilingCallbacks.shift()/*!*/[1](ev.data.err);
      } else {
        endProfilingCallbacks.shift()/*!*/[0]();
      }
      break;
    default:
  }
};

const scriptSrc = typeof document !== 'undefined' ? (document?.currentScript /*as HTMLScriptElement*/)?.src : undefined;
/**
 * @returns {Promise<void>}
 */
export const initializeWebAssemblyInstance = async() => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    if (initialized) {
      return;
    }
    if (initializing) {
      throw new Error('multiple calls to \'initWasm()\' detected.');
    }
    if (aborted) {
      throw new Error('previous call to \'initWasm()\' failed.');
    }

    initializing = true;

    // overwrite wasm filepaths
    if (env.wasm.wasmPaths === undefined) {
      if (scriptSrc && scriptSrc.indexOf('blob:') !== 0) {
        env.wasm.wasmPaths = scriptSrc.substr(0, +(scriptSrc).lastIndexOf('/') + 1);
      }
    }

    return new Promise/*<void>*/((resolve, reject) => {
      proxyWorker?.terminate();
      // eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
      proxyWorker = require('worker-loader?inline=no-fallback!./proxy-worker/main').default() /*as Worker*/;
      proxyWorker.onmessage = onProxyWorkerMessage;
      initWasmCallbacks = [resolve, reject];
      /** @type {OrtWasmMessage} */
      const message = {type: 'init-wasm', in : env.wasm};
      proxyWorker.postMessage(message);
    });

  } else {
    return initializeWebAssembly(env.wasm);
  }
};
/**
 *
 * @param {Env} env
 * @returns {Promise<void>}
 */
export const initializeRuntime = async(env) => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise/*<void>*/((resolve, reject) => {
      initOrtCallbacks = [resolve, reject];
      /** @type {OrtWasmMessage} */
      const message = {type: 'init-ort', in : env};
      proxyWorker/*!*/.postMessage(message);
    });
  } else {
    await core.initRuntime(env);
  }
};
/**
 *
 * @param {Uint8Array} model
 * @returns {Promise<SerializableModeldata>}
 */
export const createSessionAllocate = async(model) => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise<SerializableModeldata>((resolve, reject) => {
      createSessionAllocateCallbacks.push([resolve, reject]);
      /** @type {OrtWasmMessage} */
      const message = {type: 'create_allocate', in : {model}};
      proxyWorker/*!*/.postMessage(message, [model.buffer]);
    });
  } else {
    return core.createSessionAllocate(model);
  }
};
/**
 *
 * @param {SerializableModeldata} modeldata
 * @param {InferenceSession.SessionOptions} [options]
 * @returns {Promise<SerializableSessionMetadata>}
 */
export const createSessionFinalize = async(modeldata, options) => {
      if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
        ensureWorker();
        return new Promise<SerializableSessionMetadata>((resolve, reject) => {
          createSessionFinalizeCallbacks.push([resolve, reject]);
          /** @type {OrtWasmMessage} */
          const message = {type: 'create_finalize', in : {modeldata, options}};
          proxyWorker/*!*/.postMessage(message);
        });
      } else {
        return core.createSessionFinalize(modeldata, options);
      }
    };
/**
 *
 * @param {Uint8Array} model
 * @param {InferenceSession.SessionOptions} [options]
 * @returns {Promise<SerializableSessionMetadata>}
 */
export const createSession = async(model, options) => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise<SerializableSessionMetadata>((resolve, reject) => {
      createSessionCallbacks.push([resolve, reject]);
      /** @type {OrtWasmMessage} */
      const message = {type: 'create', in : {model, options}};
      proxyWorker/*!*/.postMessage(message, [model.buffer]);
    });
  } else {
    return core.createSession(model, options);
  }
};
/**
 *
 * @param {number} sessionId
 * @returns {Promise<void>}
 */
export const releaseSession = async(sessionId) => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise/*<void>*/((resolve, reject) => {
      releaseSessionCallbacks.push([resolve, reject]);
      /** @type {OrtWasmMessage} */
      const message = {type: 'release', in : sessionId};
      proxyWorker/*!*/.postMessage(message);
    });
  } else {
    core.releaseSession(sessionId);
  }
};
/**
 *
 * @param {number} sessionId
 * @param {number[]} inputIndices
 * @param {SerializableTensor[]} inputs
 * @param {number[]} outputIndices
 * @param {InferenceSession.RunOptions} options
 * @returns {Promise<SerializableTensor[]>}
 */
export const run = async(sessionId, inputIndices, inputs, outputIndices, options) => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise/*<SerializableTensor[]>*/((resolve, reject) => {
      runCallbacks.push([resolve, reject]);
      /** @type {OrtWasmMessage} */
      const message = {type: 'run', in : {sessionId, inputIndices, inputs, outputIndices, options}};
      proxyWorker/*!*/.postMessage(message, core.extractTransferableBuffers(inputs));
    });
  } else {
    return core.run(sessionId, inputIndices, inputs, outputIndices, options);
  }
};
/**
 * @param {number} sessionId
 * @returns {Promise<void>}
 */
export const endProfiling = async(sessionId) => {
  if (!BUILD_DEFS.DISABLE_WASM_PROXY && isProxy()) {
    ensureWorker();
    return new Promise/*<void>*/((resolve, reject) => {
      endProfilingCallbacks.push([resolve, reject]);
      /** @type {OrtWasmMessage} */
      const message = {type: 'end-profiling', in : sessionId};
      proxyWorker/*!*/.postMessage(message);
    });
  } else {
    core.endProfiling(sessionId);
  }
};
