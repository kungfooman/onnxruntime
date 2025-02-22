// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable no-param-reassign */

export class MatMulUtil {
  /**
   * Calculate the expected shape when matrix multiplication
   * @param {[number, number]} a The shape of tensor A. Should be a tuple of 2 positive integers
   * @param {[number, number]} b The shape of tensor B. Should be a tuple of 2 positive integers
   * @returns {[number, number]|undefined} The expected shape of the result, or undefined if N/A
   */
  static calcMatMulShape(a, b) {
    return (a[1] !== b[0]) ? undefined : [a[0], b[1]];
  }
}


export class BroadcastUtil {
  /**
   * Calculate the expected shape when broadcasting 2 tensors
   * @param {readonly number[]} adims The shape of tensor A. Should be an array of positive integers
   * @param {readonly number[]} bdims The shape of tensor B. Should be an array of positive integers
   * @param {boolean} isMatMul Whether the operation is MatMul
   * @returns {readonly number[]|undefined} The expected shape of the result, or undefined if N/A
   */
  static calcShape(adims, bdims, isMatMul = false) {
    const arank = adims.length;
    const brank = bdims.length;
    if (arank === 0) {
      return bdims;
    }
    if (brank === 0) {
      return adims;
    }
    const crank = Math.max(adims.length, bdims.length);
    const cdims = new Array<number>(crank);

    // calculate the last 2 dimension if it is MatMul
    if (isMatMul) {
      if (arank < 2 || brank < 2) {
        return undefined;
      }
      const cShapeMatMul =
          MatMulUtil.calcMatMulShape([adims[arank - 2], adims[arank - 1]], [bdims[brank - 2], bdims[brank - 1]]);
      if (cShapeMatMul === undefined) {
        return undefined;
      }
      [cdims[crank - 2], cdims[crank - 1]] = cShapeMatMul;
    }

    for (let i = isMatMul ? 3 : 1; i <= crank; i++) {
      const aLen = arank - i < 0 ? 1 : adims[arank - i];
      const bLen = brank - i < 0 ? 1 : bdims[brank - i];

      if (aLen !== bLen && aLen > 1 && bLen > 1) {
        return undefined;
      }
      cdims[crank - i] = Math.max(aLen, bLen);
    }

    return cdims;
  }

  /**
   * Determine if a shape is unidirectional broadcastable to another shape
   * @param {readonly number[]} shape The input shape
   * @param {readonly number[]} finalShape The desired shape after broadcasting
   * @returns {boolean}
   */
  static isValidBroadcast(shape, finalShape) {
    // align shape to the right
    const inputRank = shape.length;
    const finalRank = finalShape.length;
    if (inputRank > finalRank) {
      return false;
    }
    for (let i = 1; i <= inputRank; i++) {
      if (shape[inputRank - i] !== 1 && shape[inputRank - i] !== finalShape[finalRank - i]) {
        return false;
      }
    }
    return true;
  }
}


export class ShapeUtil {
  /**
   * calculate the size (number of elements)
   * @param {readonly number[]} dims
   * @returns {number}
   */
  static size(dims) {
    return ShapeUtil.getSizeFromDimensionRange(dims, 0, dims.length);
  }

  /**
   * calculate the size (number of elements) from the given axis (inclusive)
   * @param {readonly number[]} dims
   * @param {number} axis
   * @throws {TypeError}
   * @returns {number}
   */
  static sizeFromDimension(dims, axis) {
    if (axis < 0 || axis > dims.length) {
      throw new TypeError(`invalid dimension of ${axis} for sizeFromDimension as Tensor has ${dims.length} dimensions.`);
    }
    return ShapeUtil.getSizeFromDimensionRange(dims, axis, dims.length);
  }

  /**
   * calculate the size (number of elements) to the given axis (exclusive)
   * @param {readonly number[]} dims
   * @param {number} axis
   * @returns {number}
   */
  static sizeToDimension(dims, axis) {
    if (axis < 0 || axis > dims.length) {
      throw new Error(`invalid dimension of ${axis} for sizeToDimension as Tensor has ${dims.length} dimensions.`);
    }
    return ShapeUtil.getSizeFromDimensionRange(dims, 0, axis);
  }

  /**
   * calculate the size (number of elements) from and to the given axis [start, end)
   * @param {readonly number[]} dims
   * @param {number} start
   * @param {number} end
   * @returns {number}
   */
  static getSizeFromDimensionRange(dims, start, end) {
    let size = 1;
    for (let i = start; i < end; i++) {
      // safety check as this method is called by multiple other methods requiring size.
      // size cannot be 0 or negative.
      if (dims[i] <= 0) {
        throw new Error(
            // eslint-disable-next-line max-len
            'cannot get valid size from specified dimension range. Most likely the range contains 0 or negative values in them.');
      }
      size *= dims[i];
    }
    return size;
  }

  /**
   * @param {readonly number[]} dims
   * @returns {readonly number[]}
   */
  static computeStrides(dims) {
    const rank = dims.length;
    if (rank === 0) {
      return [];
    } else if (rank === 1) {
      return [1];
    }
    const strides = new Array(rank);
    strides[rank - 1] = 1;
    strides[rank - 2] = dims[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
  }

  /**
   * normalize axis of range [-r, r) into [0, r).
   * @param {number} axis
   * @param {number} tensorRank
   * @returns {number}
   */
  static normalizeAxis(axis, tensorRank) {
    if (axis < -tensorRank && axis >= tensorRank) {
      throw new Error('unsupported axis for this operation.');
    }
    return axis < 0 ? axis + tensorRank : axis;
  }

  /**
   * @param {readonly number[]} axes
   * @param {number} [tensorRank]
   * @returns {number[]}
   */
  static normalizeAxes(axes, tensorRank) {
    return axes.map(x => this.normalizeAxis(x, tensorRank ?? axes.length));
  }

  /**
   * Sorts a given array based on the indices in the Perm array
   * Used in Transpose
   * @param {readonly number[]} a Array to be sorted such as dims or strides
   * @param {readonly number[]} [perm] Perm given; if null a will be reversed
   * @returns {readonly number[]}
   */
  static sortBasedOnPerm(a, perm) {
    if (perm) {
      return perm.map((v) => a[v]);
    } else {
      return a.slice().reverse();
    }
  }

  /**
   * Pads a given shape according to the padding values
   * @param {readonly number[]} dims shape of the Tensor to be padded
   * @param {readonly number[]} pad pad values
   * @returns {readonly number[]}
   */
  static padShape(dims, pad) {
    const rank = dims.length;
    return dims.map((v, i) => v + pad[i] + pad[i + rank]);
  }

  /**
   * Determines if the two shapes are identical
   * @param {readonly number[]} shape1
   * @param {readonly number[]} shape2
   * @returns {boolean}
   */
  static areEqual(shape1, shape2) {
    if (shape1.length !== shape2.length) {
      return false;
    }
    return shape1.every((v, i) => v === shape2[i]);
  }
}

export class PoolConvUtil {
  /**
   * Adjust the kernel, strides, pads to correct rank. Set to default value if not present
   * @param {boolean} isGlobalOperator If true, perform global pooling.
   * @param {readonly number[]} inputDims The input tensor dimension.
   * @param {number[]} kernelShape The size of the kernel along each axis.
   * @param {number[]} strides Stride along each axis.
   * @param {number[]} dilations Dilation along each axis.
   * @param {number[]} pads Padding for the beginning and ending along each axis.
   * @returns {void}
   */
  static adjustPoolAttributes(isGlobalOperator, inputDims, kernelShape, strides, dilations, pads) {
    if (!isGlobalOperator && kernelShape.length !== inputDims.length - 2) {
      throw new Error('length of specified kernel shapes should be 2 less than length of input dimensions');
    }

    if (isGlobalOperator) {
      // adjust kernel shape to cover the input dims
      for (let dim = 0; dim < inputDims.length - 2; dim++) {
        if (dim >= kernelShape.length) {
          kernelShape.push(inputDims[dim + 2]);
        } else {
          kernelShape[dim] = inputDims[dim + 2];
        }
      }
    }

    // adjust strides length to match kernel shape length
    for (let dim = 0; dim < kernelShape.length; dim++) {
      if (dim < strides.length) {
        if (strides[dim] < 0) {
          throw new Error('strides should be greater than or equal to 1');
        }
      } else {
        strides.push(1);
      }
    }

    // adjust dilation value
    for (let dim = 0; dim < kernelShape.length; dim++) {
      if (dim < dilations.length) {
        if (dilations[dim] < 0) {
          throw new Error('dilations should be greater than or equal to 1');
        }
      } else {
        dilations.push(1);
      }
    }

    // adjust pads length to match 2 * kernel shape length
    for (let dim = 0; dim < kernelShape.length * 2; dim++) {
      if (dim < pads.length) {
        if (pads[dim] < 0) {
          throw new Error('pad should be greater than or equal to 1');
        }
      } else {
        pads.push(0);
      }
    }

    // sanity checks for values in kernel shapes and pads
    for (let dim = 0; dim < kernelShape.length; dim++) {
      if (kernelShape[dim] <= 0) {
        throw new Error('kernel shapes need to be greater than 0');
      }

      if (pads[dim] >= kernelShape[dim] || pads[dim + kernelShape.length] >= kernelShape[dim]) {
        throw new Error('pads should be smaller than kernel');
      }
    }
  }

  /**
   * Adjust pad values based on 'autoPad' attribute.
   *
   * @param {readonly number[]} inputDims
   * @param {readonly number[]} strides
   * @param {readonly number[]} dilations
   * @param {readonly number[]} kernelShape
   * @param {number[]} pads
   * @param {boolean} isChannelLast
   * @param {string} [autoPad]
   * @throws {Error}
   * @returns {void}
   */
  static adjustPadsBasedOnAutoPad(inputDims, strides, dilations, kernelShape, pads, isChannelLast, autoPad) {
    if (!autoPad) {
      return;
    }

    if (pads.length !== 2 * (inputDims.length - 2)) {
      throw new Error('length of pads should be twice the length of data dimensions');
    }

    if (strides.length !== (inputDims.length - 2)) {
      throw new Error('length of strides should be the length of data dimensions');
    }

    if (kernelShape.length !== (inputDims.length - 2)) {
      throw new Error('length of kernel shapes should be the length of data dimensions');
    }

    for (let dim = 0; dim < inputDims.length - 2; dim++) {
      PoolConvUtil.adjustPadAndReturnShape(
          inputDims[dim + (isChannelLast ? 1 : 2)], strides[dim], dilations[dim], kernelShape[dim], pads, dim,
          dim + inputDims.length - 2, autoPad);
    }
  }

  /**
   * Calculate the output shape for Pool ops based on input attributes. (Should be used only for Pool ops)
   * @param {boolean} isGlobalOperator If true, perform global pooling.
   * @param {readonly number[]} inputDims The input tensor dimension. (inputs[0].dims)
   * @param {number[]} strides Stride along each axis.
   * @param {number[]} dilations Dilation along each axis.
   * @param {number[]} kernelShape The size of the kernel along each axis.
   * @param {number[]} pads Padding for the beginning and ending along each axis.
   * @param {string} [autoPad] DEPRECATED attribute supported for legacy models. Specifies
   * how to implicitly calculate pads in each dimension. Can take values NOTSET, SAME_UPPER,
   * SAME_LOWER, or VALID.
   * @throws {Error}
   * @returns {number[]}
   */
  static computePoolOutputShape(isGlobalOperator, inputDims, strides, dilations, kernelShape, pads, autoPad) {
    if (inputDims.length <= 0) {
      throw new Error('input shape must be of size greater than 0');
    }
    // Add batch size and number of channels of output
    const outputDims = [inputDims[0], inputDims[1]];
    PoolConvUtil.computeShapeHelper(
      isGlobalOperator, inputDims, outputDims, strides, dilations, kernelShape, pads, autoPad
    );
    return outputDims;
  }

  /**
   * Calculate the output shape for Conv op based on input attributes. (Should be used only for Conv op)
   * @param {readonly number[]} inputDims The input tensor dimension. (inputs[0].dims)
   * @param {readonly number[]} filterDims The filter tensor dimension. (inputs[1].dims)
   * @param {number[]} strides Stride along each axis.
   * @param {number[]} dilations
   * @param {number[]} kernelShape The size of the kernel along each axis.
   * @param {number[]} pads Padding for the beginning and ending along each axis.
   * @param {string} [autoPad] DEPRECATED attribute supported for legacy models. Specifies how
   * to implicitly calculate pads in each dimension. Can take values NOTSET, SAME_UPPER,
   * SAME_LOWER, or VALID.
   * @returns {number[]}
   */
  static computeConvOutputShape(inputDims, filterDims, strides, dilations, kernelShape, pads, autoPad) {
    if (inputDims.length <= 0 || filterDims.length <= 0) {
      throw new Error('invalid input tensor dims or invalid filter tensor dims');
    }
    // Add batch size and number of channels of output
    const outputDims = [inputDims[0], filterDims[0]];
    PoolConvUtil.computeShapeHelper(false, inputDims, outputDims, strides, dilations, kernelShape, pads, autoPad);
    return outputDims;
  }

  /**
   * will compute output shapes for data dimensions ONLY (i.e.) no batch size and channels
   * called by computePoolOutputShape() and computeConvOutputShape()
   * adjust pads based on 'autoPad' attribute prior to shape computation
   *
   * @param {boolean} isGlobalOperator
   * @param {readonly number[]} inputDims
   * @param {number[]} outputDims
   * @param {readonly number[]} strides
   * @param {readonly number[]} dilations
   * @param {readonly number[]} kernelShape
   * @param {number[]} pads
   * @param {string} [autoPad]
   * @private
   */
  static computeShapeHelper(
    isGlobalOperator, inputDims, outputDims, strides, dilations, kernelShape, pads, autoPad
  ) {
    if (isGlobalOperator) {
      for (let dim = 0; dim < inputDims.length - 2; dim++) {
        outputDims.push(1);
      }
    } else {
      for (let dim = 0; dim < inputDims.length - 2; dim++) {
        outputDims.push(PoolConvUtil.adjustPadAndReturnShape(
            inputDims[dim + 2], strides[dim], dilations[dim], kernelShape[dim], pads, dim, dim + inputDims.length - 2,
            autoPad));
      }
    }
  }

  /**
   * helper for computeShapeHelper() and adjustPadsBasedOnAutoPad()
   * adjusts pad value for given 'autoPad' string and computes output shape along a particular dimension
   * @param {number} inSize
   * @param {number} stride
   * @param {number} dilation
   * @param {number} kernel
   * @param {number[]} pads
   * @param {number} padHeadIndex
   * @param {number} padTailIndex
   * @param {string} [autoPad]
   * @private
   * @returns {number}
   */
  static adjustPadAndReturnShape(
    inSize, stride, dilation, kernel, pads, padHeadIndex, padTailIndex, autoPad
  ) {
    const dkernel = dilation * (kernel - 1) + 1;
    if (autoPad && autoPad !== 'NOTSET') {
      switch (autoPad) {
        case 'VALID':
          pads[padHeadIndex] = 0;
          pads[padTailIndex] = 0;
          return Math.floor(((inSize - dkernel) / stride) + 1);
        case 'SAME_LOWER':
        case 'SAME_UPPER':
          if (dilation !== 1) {
            throw new Error('Dilation not supported for SAME_UPPER or SAME_LOWER');
          } else {
            const legacyTargetSize = (inSize + stride - 1) / stride;
            const padNeeded = (legacyTargetSize - 1) * stride + kernel - inSize;
            pads[padHeadIndex] =
                (autoPad === 'SAME_LOWER') ? Math.floor((padNeeded + 1) / 2) : Math.floor(padNeeded / 2);
            pads[padTailIndex] = padNeeded - pads[padHeadIndex];
            return Math.floor(((inSize + padNeeded - kernel) / stride) + 1);
          }
        default:
          throw new Error('Unsupported AutoPad type');
      }
    } else {
      return Math.floor(((inSize + pads[padHeadIndex] + pads[padTailIndex] - dkernel) / stride) + 1);
    }
  }
}

export class GemmUtil {
  /**
   * will make sure input shapes are compatible for this op
   * and return back the shape of the output in the form of a tuple
   * will throw exception if the input shapes are not compatible
   *
   * @param {readonly number[]} leftShape
   * @param {boolean} transLeft
   * @param {readonly number[]} rightShape
   * @param {boolean} transRight
   * @param {readonly number[]} [biasShape]
   * @returns {readonly number[]}
   */
  static getShapeOfGemmResult(
    leftShape, transLeft, rightShape, transRight, biasShape
  ) {
    if (leftShape.length !== 2 || rightShape.length !== 2) {
      throw new Error('shape need to be of size 2');
    }

    /** @type {number} */
    let M;
    /** @type {number} */
    let K;
    /** @type {number} */
    let N;

    if (transLeft) {
      M = leftShape[1];
      K = leftShape[0];
    } else {
      M = leftShape[0];
      K = leftShape[1];
    }

    let kDim = -1;

    if (transRight) {
      N = rightShape[0];
      kDim = 1;
    } else {
      N = rightShape[1];
      kDim = 0;
    }

    if (rightShape[kDim] !== K) {
      throw new Error('dimension mismatch');
    }

    if (M <= 0 || N <= 0 || K <= 0) {
      throw new Error('invalid shape specified');
    }

    if (biasShape && !BroadcastUtil.isValidBroadcast(biasShape, [M, N])) {
      throw new Error('gemm: invalid bias shape for broadcast');
    }

    return [M, N, K];
  }
}


export const MIN_CLIP = -3.4028234663852886e+38;
export const MAX_CLIP = 3.4028234663852886e+38;
