// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//import {SessionHandler} from './backend';
import {resolveBackend} from './backend.js';
//import {InferenceSession as InferenceSessionInterface} from './inference-session.js';
//import {OnnxValue} from './onnx-value';
import {Tensor} from './tensor.js';

//type SessionOptions = InferenceSessionInterface.SessionOptions;
//type RunOptions = InferenceSessionInterface.RunOptions;
//type FeedsType = InferenceSessionInterface.FeedsType;
//type FetchesType = InferenceSessionInterface.FetchesType;
//type ReturnType = InferenceSessionInterface.ReturnType;

/**
 * @implements {InferenceSessionInterface}
 */

export class InferenceSession {
  /**
   * @param {SessionHandler} handler
   * @private
   */
  constructor(handler) {
    this.handler = handler;
  }
  //run(feeds: FeedsType, options?: RunOptions): Promise<ReturnType>;
  //run(feeds: FeedsType, fetches: FetchesType, options?: RunOptions): Promise<ReturnType>;
  /**
   *
   * @param {FeedsType} feeds
   * @param {FetchesType|RunOptions} [arg1]
   * @param {RunOptions} [arg2]
   * @returns {Promise<ReturnType>}
   */
  async run(feeds, arg1, arg2) {
    /** @type {{[name: string]: OnnxValue|null}} */
    const fetches = {};
    /** @type {RunOptions} */
    let options = {};
    // check inputs
    if (typeof feeds !== 'object' || feeds === null || feeds instanceof Tensor || Array.isArray(feeds)) {
      throw new TypeError(
          '\'feeds\' must be an object that use input names as keys and OnnxValue as corresponding values.');
    }

    let isFetchesEmpty = true;
    // determine which override is being used
    if (typeof arg1 === 'object') {
      if (arg1 === null) {
        throw new TypeError('Unexpected argument[1]: cannot be null.');
      }
      if (arg1 instanceof Tensor) {
        throw new TypeError('\'fetches\' cannot be a Tensor');
      }

      if (Array.isArray(arg1)) {
        if (arg1.length === 0) {
          throw new TypeError('\'fetches\' cannot be an empty array.');
        }
        isFetchesEmpty = false;
        // output names
        for (const name of arg1) {
          if (typeof name !== 'string') {
            throw new TypeError('\'fetches\' must be a string array or an object.');
          }
          if (this.outputNames.indexOf(name) === -1) {
            throw new RangeError(`'fetches' contains invalid output name: ${name}.`);
          }
          fetches[name] = null;
        }

        if (typeof arg2 === 'object' && arg2 !== null) {
          options = arg2;
        } else if (typeof arg2 !== 'undefined') {
          throw new TypeError('\'options\' must be an object.');
        }
      } else {
        // decide whether arg1 is fetches or options
        // if any output name is present and its value is valid OnnxValue, we consider it fetches
        let isFetches = false;
        const arg1Keys = Object.getOwnPropertyNames(arg1);
        for (const name of this.outputNames) {
          if (arg1Keys.indexOf(name) !== -1) {
            const v = (arg1 /*as InferenceSessionInterface.NullableOnnxValueMapType*/)[name];
            if (v === null || v instanceof Tensor) {
              isFetches = true;
              isFetchesEmpty = false;
              fetches[name] = v;
            }
          }
        }

        if (isFetches) {
          if (typeof arg2 === 'object' && arg2 !== null) {
            options = arg2;
          } else if (typeof arg2 !== 'undefined') {
            throw new TypeError('\'options\' must be an object.');
          }
        } else {
          options = arg1 /*as RunOptions*/;
        }
      }
    } else if (typeof arg1 !== 'undefined') {
      throw new TypeError('Unexpected argument[1]: must be \'fetches\' or \'options\'.');
    }

    // check if all inputs are in feed
    for (const name of this.inputNames) {
      if (typeof feeds[name] === 'undefined') {
        throw new Error(`input '${name}' is missing in 'feeds'.`);
      }
    }

    // if no fetches is specified, we use the full output names list
    if (isFetchesEmpty) {
      for (const name of this.outputNames) {
        fetches[name] = null;
      }
    }

    // feeds, fetches and options are prepared

    const results = await this.handler.run(feeds, fetches, options);
    /** @type {{[name: string]: OnnxValue}} */
    const returnValue = {};
    for (const key in results) {
      if (Object.hasOwnProperty.call(results, key)) {
        returnValue[key] = new Tensor(results[key].type, results[key].data, results[key].dims);
      }
    }
    return returnValue;
  }

  //static create(path: string, options?: SessionOptions): Promise<InferenceSessionInterface>;
  //static create(buffer: ArrayBufferLike, options?: SessionOptions): Promise<InferenceSessionInterface>;
  //static create(buffer: ArrayBufferLike, byteOffset: number, byteLength?: number, options?: SessionOptions):
  //    Promise<InferenceSessionInterface>;
  //static create(buffer: Uint8Array, options?: SessionOptions): Promise<InferenceSessionInterface>;
  /**
   *
   * @param {string|ArrayBufferLike|Uint8Array} arg0
   * @param {SessionOptions|number} [arg1]
   * @param {number} [arg2]
   * @param {SessionOptions} [arg3]
   * @returns {Promise<InferenceSessionInterface>}
   */
  static async create(arg0, arg1, arg2, arg3) {
    // either load from a file or buffer
    /** @type {string|Uint8Array} */
    let filePathOrUint8Array;
    /** @type {SessionOptions} */
    let options = {};

    if (typeof arg0 === 'string') {
      filePathOrUint8Array = arg0;
      if (typeof arg1 === 'object' && arg1 !== null) {
        options = arg1;
      } else if (typeof arg1 !== 'undefined') {
        throw new TypeError('\'options\' must be an object.');
      }
    } else if (arg0 instanceof Uint8Array) {
      filePathOrUint8Array = arg0;
      if (typeof arg1 === 'object' && arg1 !== null) {
        options = arg1;
      } else if (typeof arg1 !== 'undefined') {
        throw new TypeError('\'options\' must be an object.');
      }
    } else if (
        arg0 instanceof ArrayBuffer ||
        (typeof SharedArrayBuffer !== 'undefined' && arg0 instanceof SharedArrayBuffer)) {
      const buffer = arg0;
      let byteOffset = 0;
      let byteLength = arg0.byteLength;
      if (typeof arg1 === 'object' && arg1 !== null) {
        options = arg1;
      } else if (typeof arg1 === 'number') {
        byteOffset = arg1;
        if (!Number.isSafeInteger(byteOffset)) {
          throw new RangeError('\'byteOffset\' must be an integer.');
        }
        if (byteOffset < 0 || byteOffset >= buffer.byteLength) {
          throw new RangeError(`'byteOffset' is out of range [0, ${buffer.byteLength}).`);
        }
        byteLength = arg0.byteLength - byteOffset;
        if (typeof arg2 === 'number') {
          byteLength = arg2;
          if (!Number.isSafeInteger(byteLength)) {
            throw new RangeError('\'byteLength\' must be an integer.');
          }
          if (byteLength <= 0 || byteOffset + byteLength > buffer.byteLength) {
            throw new RangeError(`'byteLength' is out of range (0, ${buffer.byteLength - byteOffset}].`);
          }
          if (typeof arg3 === 'object' && arg3 !== null) {
            options = arg3;
          } else if (typeof arg3 !== 'undefined') {
            throw new TypeError('\'options\' must be an object.');
          }
        } else if (typeof arg2 !== 'undefined') {
          throw new TypeError('\'byteLength\' must be a number.');
        }
      } else if (typeof arg1 !== 'undefined') {
        throw new TypeError('\'options\' must be an object.');
      }
      filePathOrUint8Array = new Uint8Array(buffer, byteOffset, byteLength);
    } else {
      throw new TypeError('Unexpected argument[0]: must be \'path\' or \'buffer\'.');
    }

    // get backend hints
    const eps = options.executionProviders || [];
    const backendHints = eps.map(i => typeof i === 'string' ? i : i.name);
    const backend = await resolveBackend(backendHints);
    const handler = await backend.createSessionHandler(filePathOrUint8Array, options);
    return new InferenceSession(handler);
  }

  startProfiling() {
    this.handler.startProfiling();
  }
  endProfiling() {
    this.handler.endProfiling();
  }

  /**
   * @type {readonly string[]}
   */
  get inputNames() {
    return this.handler.inputNames;
  }

  /**
   * @type {readonly string[]}
   */
  get outputNames() {
    return this.handler.outputNames;
  }

  /**
   * @type {SessionHandler}
   * @private
   */
  handler;
}
