// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {WebGpuBackend} from '../backend-webgpu.js';
import {LOG_DEBUG} from '../log.js';

import {
  // GpuData,
  // GpuDataId,
  GpuDataType
} from './types.js';

/**
 * manages GpuDataId -> GpuBuffer
 *
 * @interface
 */
export class GpuDataManager {
  /**
   * copy data from CPU to GPU.
   * @param {GpuDataId}
   * @param {Uint8Array}
   * @returns {void}
   */
  upload(id, data) {
    throw "not implemented";
  }
  /**
   * copy data from GPU to GPU.
   * @param {GpuDataId} sourceId
   * @param {GpuDataId} destinationId
   * @returns {void}
   */
  memcpy(sourceId, destinationId) {
    throw "not implemented";
  }
  /**
   * create new data on GPU.
   * @param {number} size
   * @param {number} [usage]
   * @returns {GpuData}
   */
  create(size, usage) {
    throw "not implemented";
  }
  /**
   * get GPU data by ID.
   * @param {GpuDataId} id
   * @param {}
   * @returns {GpuData|undefined}
   */
  get(id) {
    throw "not implemented";
  }
  /**
   * release the data on GPU by ID.
   * @param {GpuDataId} id
   * @returns {number} size of the data released
   */
  release(id) {
    throw "not implemented";
  }
  /**
   * copy data from GPU to CPU.
   * @param {GpuDataId} id
   * @returns {Promise<ArrayBufferLike>}
   */
  download(id) {
    throw "not implemented";
  }

  /**
   * refresh the buffers that marked for release.
   *
   * when release() is called, the buffer is not released immediately. this is because we need to wait for the commands
   * to be submitted to the GPU. this function is called after the commands are submitted so that the buffers can be
   * actually released.
   * @returns {void}
   */
  refreshPendingBuffers() {
    throw "not implemented";
  }
}

/**
 * @typedef {object} StorageCacheValue
 * @property {GpuData} gpuData
 * @property {number} originalSize
 */

/**
 * @typedef {object} DownloadCacheValue
 * @property {Promise<ArrayBufferLike>} data
 */

/**
 * normalize the buffer size so that it fits the 128-bits (16 bytes) alignment.
 *
 * @param {number} size
 */
const calcNormalizedBufferSize = (size) => Math.ceil(size / 16) * 16;

let guid = 0;
const createNewGpuDataId = () => guid++;

class GpuDataManagerImpl extends GpuDataManager {
  /**
   * GPU Data ID => GPU Data ( storage buffer )
   * @type {Map<GpuDataId, StorageCacheValue>}
   * */
  storageCache;
  /**
   * GPU Data ID => GPU Data ( read buffer )
   * @type {Map<GpuDataId, DownloadCacheValue>}
   * */
  downloadCache;
  /**
   * pending buffers for uploading ( data is unmapped )
   * @type {GPUBuffer[]}
   * @private
   */
  buffersForUploadingPending;
  /**
   * pending buffers for computing
   * @type {GPUBuffer[]}
   * @private
   */
  buffersPending;
  /**
   * @type {WebGpuBackend}
   * @private
   */
  backend;
  /**
   *
   * @param {WebGpuBackend} backend
   */
  constructor(backend /* , private reuseBuffer: boolean */) {
    super(backend);
    this.backend = backend;
    this.storageCache = new Map();
    this.downloadCache = new Map();
    this.buffersForUploadingPending = [];
    this.buffersPending = [];
  }

  /**
   * @param {GpuDataId} id
   * @param {Uint8Array} data
   * @returns {void}
   */
  upload(id, data) {
    const srcArrayBuffer = data.buffer;
    const srcOffset = data.byteOffset;
    const srcLength = data.byteLength;
    const size = calcNormalizedBufferSize(srcLength);

    // get destination gpu buffer
    const gpuDataCache = this.storageCache.get(id);
    if (!gpuDataCache) {
      throw new Error('gpu data for uploading does not exist');
    }
    if (gpuDataCache.originalSize !== srcLength) {
      throw new Error(`inconsistent data size. gpu data size=${gpuDataCache.originalSize}, data size=${srcLength}`);
    }

    // create gpu buffer
    const gpuBufferForUploading = this.backend.device.createBuffer(
        // eslint-disable-next-line no-bitwise
        {mappedAtCreation: true, size, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC});

    // copy (upload) data
    const arrayBuffer = gpuBufferForUploading.getMappedRange();
    new Uint8Array(arrayBuffer).set(new Uint8Array(srcArrayBuffer, srcOffset, srcLength));
    gpuBufferForUploading.unmap();


    // GPU copy
    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    commandEncoder.copyBufferToBuffer(gpuBufferForUploading, 0, gpuDataCache.gpuData.buffer, 0, size);

    LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.upload(id=${id})`);

    this.buffersForUploadingPending.push(gpuBufferForUploading);
  }
  /**
   *
   * @param {GpuDataId} sourceId
   * @param {GpuDataId} destinationId
   * @returns {void}
   */
  memcpy(sourceId, destinationId) {
    // get source gpu buffer
    const sourceGpuDataCache = this.storageCache.get(sourceId);
    if (!sourceGpuDataCache) {
      throw new Error('source gpu data for memcpy does not exist');
    }
    // get destination gpu buffer
    const destinationGpuDataCache = this.storageCache.get(destinationId);
    if (!destinationGpuDataCache) {
      throw new Error('destination gpu data for memcpy does not exist');
    }
    if (sourceGpuDataCache.originalSize !== destinationGpuDataCache.originalSize) {
      throw new Error('inconsistent source and destination gpu data size');
    }
    const size = calcNormalizedBufferSize(sourceGpuDataCache.originalSize);
    // GPU copy
    this.backend.getCommandEncoder().copyBufferToBuffer(
        sourceGpuDataCache.gpuData.buffer, 0, destinationGpuDataCache.gpuData.buffer, 0, size);
  }

  // eslint-disable-next-line no-bitwise
  /**
   *
   * @param {number} size
   * @param {*} usage
   * @returns {GpuData}
   */
  create(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    // !!!
    // !!! IMPORTANT: TODO: whether we should keep the storage buffer every time, or always create new ones.
    // !!!                  This need to be figured out by performance test results.
    // !!!

    const bufferSize = calcNormalizedBufferSize(size);

    // create gpu buffer
    const gpuBuffer = this.backend.device.createBuffer({size: bufferSize, usage});

    const gpuData = {id: createNewGpuDataId(), type: GpuDataType.default, buffer: gpuBuffer};
    this.storageCache.set(gpuData.id, {gpuData, originalSize: size});

    LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.create(size=${size}) => id=${gpuData.id}`);
    return gpuData;
  }
  /**
   *
   * @param {GpuDataId} id
   * @returns {GpuData|undefined}
   */
  get(id) {
    return this.storageCache.get(id)?.gpuData;
  }
  /**
   * @param {GpuDataId} id
   * @returns {number}
   */
  release(id) {
    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('releasing data does not exist');
    }

    LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.release(id=${id}), gpuDataId=${cachedData.gpuData.id}`);

    this.storageCache.delete(id);
    this.buffersPending.push(cachedData.gpuData.buffer);
    // cachedData.gpuData.buffer.destroy();

    const downloadingData = this.downloadCache.get(id);
    if (downloadingData) {
      this.downloadCache.delete(id);
    }

    return cachedData.originalSize;
  }
  /**
   *
   * @param {GpuDataId} id
   * @returns {Promise<ArrayBufferLike>}
   */
  async download(id) {
    const downloadData = this.downloadCache.get(id);
    if (downloadData) {
      return downloadData.data;
    }

    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('data does not exist');
    }

    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    const gpuReadBuffer = this.backend.device.createBuffer(
        // eslint-disable-next-line no-bitwise
        {size: cachedData.originalSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    commandEncoder.copyBufferToBuffer(
        cachedData.gpuData.buffer /* source buffer */, 0 /* source offset */, gpuReadBuffer /* destination buffer */,
        0 /* destination offset */, cachedData.originalSize /* size */
    );
    this.backend.flush();

    const readDataPromise = new Promise<ArrayBuffer>((resolve) => {
      gpuReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const data = gpuReadBuffer.getMappedRange().slice(0);
        gpuReadBuffer.destroy();
        resolve(data);
      });
    });

    this.downloadCache.set(id, {data: readDataPromise});

    return readDataPromise;
  }
  /**
   * @returns {void}
   */
  refreshPendingBuffers() {
    for (const buffer of this.buffersForUploadingPending) {
      buffer.destroy();
    }
    for (const buffer of this.buffersPending) {
      buffer.destroy();
    }
  }
}
/**
 *
 * @param  {ConstructorParameters<typeof GpuDataManagerImpl>} args
 * @returns {GpuDataManager}
 */
export const createGpuDataManager = (...args) => new GpuDataManagerImpl(...args);
