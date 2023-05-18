// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {readFile} from 'fs';
import {env, InferenceSession,
  //SessionHandler,
  Tensor} from 'onnxruntime-common';
import {promisify} from 'util';

//import {SerializableModeldata} from './proxy-messages.js';
import {createSession, createSessionAllocate, createSessionFinalize, endProfiling, initializeRuntime, releaseSession, run} from './proxy-wrapper.js';
/** @type {boolean} */
let runtimeInitialized;
/**
 * @implements {SessionHandler}
 */
export class OnnxruntimeWebAssemblySessionHandler {
  /**
   * @type {number}
   * @private
   */
  sessionId;
  /** @type {string[]} */
  inputNames;
  /** @type {string[]} */
  outputNames;
  /**
   * @param {string} path
   * @returns {Promise<SerializableModeldata>}
   */
  async createSessionAllocate(path) {
    // fetch model from url and move to wasm heap. The arraybufffer that held the http
    // response is freed once we return
    const response = await fetch(path);
    const arrayBuffer = await response.arrayBuffer();
    return createSessionAllocate(new Uint8Array(arrayBuffer));
  }
  /**
   *
   * @param {string|Uint8Array} pathOrBuffer
   * @param {InferenceSession.SessionOptions} [options]
   * @returns {Promise<void>}
   */
  async loadModel(pathOrBuffer, options) {
    if (!runtimeInitialized) {
      await initializeRuntime(env);
      runtimeInitialized = true;
    }

    if (typeof pathOrBuffer === 'string') {
      if (typeof fetch === 'undefined') {
        // node
        const model = await promisify(readFile)(pathOrBuffer);
        [this.sessionId, this.inputNames, this.outputNames] = await createSession(model, options);
      } else {
        // browser
        // fetch model and move to wasm heap.
        /** @type {SerializableModeldata} */
        const modelData = await this.createSessionAllocate(pathOrBuffer);
        // create the session
        [this.sessionId, this.inputNames, this.outputNames] = await createSessionFinalize(modelData, options);
      }
    } else {
      [this.sessionId, this.inputNames, this.outputNames] = await createSession(pathOrBuffer, options);
    }
  }
  /**
   * @returns {Promise<void>}
   */
  async dispose() {
    return releaseSession(this.sessionId);
  }
  /**
   * @param {SessionHandler.FeedsType} feeds
   * @param {SessionHandler.FetchesType} fetches
   * @param {InferenceSession.RunOptions} options
   * @returns {Promise<SessionHandler.ReturnType>}
   */
  async run(feeds, fetches, options) {
    /** @type {Tensor[]} */
    const inputArray = [];
    /** @type {number[]} */
    const inputIndices = [];
    Object.entries(feeds).forEach(kvp => {
      const name = kvp[0];
      const tensor = kvp[1];
      const index = this.inputNames.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid input '${name}'`);
      }
      inputArray.push(tensor);
      inputIndices.push(index);
    });
    /** @type {number[]} */
    const outputIndices = [];
    Object.entries(fetches).forEach(kvp => {
      const name = kvp[0];
      // TODO: support pre-allocated output
      const index = this.outputNames.indexOf(name);
      if (index === -1) {
        throw new Error(`invalid output '${name}'`);
      }
      outputIndices.push(index);
    });

    const outputs =
        await run(this.sessionId, inputIndices, inputArray.map(t => [t.type, t.dims, t.data]), outputIndices, options);
    /** @type {SessionHandler.ReturnType} */
    const result = {};
    for (let i = 0; i < outputs.length; i++) {
      result[this.outputNames[outputIndices[i]]] = new Tensor(outputs[i][0], outputs[i][2], outputs[i][1]);
    }
    return result;
  }
  /**
   * @returns {void}
   */
  startProfiling() {
    // TODO: implement profiling
  }
  /**
   * @returns {void}
   */
  endProfiling() {
    void endProfiling(this.sessionId);
  }
}
