// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/dml/DmlExecutionProvider/inc/MLOperatorAuthor.h"
#include "core/common//common.h"

struct ReluShapeInferrer : winrt::implements<ReluShapeInferrer, IMLOperatorShapeInferrer>
{
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
    {
        uint32_t inputDimsSize;
        context->GetInputTensorDimensionCount(0, &inputDimsSize);
        
        auto inputDims = new uint32_t[inputDimsSize];
        context->GetInputTensorShape(0, inputDimsSize, inputDims);
        
        context->SetOutputTensorShape(0, inputDimsSize, inputDims);
        return S_OK;
    }
};

struct ReluOperator: winrt::implements<ReluOperator, IMLOperatorKernel>
{
    ReluOperator() {}

    // Computes the outputs of the kernel.  In this case, the output will represent
    // the Rectified Linear Unit (Relu) output.
    //
    // Based on the operators location in the model graph this operator may be called multiple times
    // or simultaneously within the same instance of the class during evaluation.  Implementations
    // of this method must be thread-safe.
    STDMETHOD(Compute)(IMLOperatorKernelContext* context)
    {
        // Get the input tensor
        winrt::com_ptr<IMLOperatorTensor> inputTensor;
        context->GetInputTensor(0, inputTensor.put());

        // Get the output tensor
        winrt::com_ptr<IMLOperatorTensor> outputTensor;
        context->GetOutputTensor(0, outputTensor.put());

        // Get the input and output shape sizes
        uint32_t inputDimsSize = inputTensor->GetDimensionCount();
        uint32_t outputDimsSize = outputTensor->GetDimensionCount();
        if (inputDimsSize != outputDimsSize)
        {
            return E_UNEXPECTED;
        }

        // Get the input shape
        std::vector<uint32_t> inputDims(inputDimsSize);
        outputTensor->GetShape(inputDimsSize, inputDims.data());

        // Get the output shape
        std::vector<uint32_t> outputDims(outputDimsSize);
        outputTensor->GetShape(outputDimsSize, outputDims.data());

        // For the number of total elements in the input and output shapes
        auto outputDataSize = std::accumulate(outputDims.begin(), outputDims.end(), 1, std::multiplies<uint32_t>());
        auto inputDataSize = std::accumulate(inputDims.begin(), inputDims.end(), 1, std::multiplies<uint32_t>());
        if (outputDataSize != inputDataSize)
        {
            return E_UNEXPECTED;
        }

        // If the tensor types are both float type
        if (outputTensor->GetTensorDataType() == MLOperatorTensorDataType::Float &&
            inputTensor->GetTensorDataType() == MLOperatorTensorDataType::Float)
        {
            // For cpu data
            if (outputTensor->IsCpuData() && inputTensor->IsCpuData())
            {
                ComputeInternal<float>(inputTensor.get(), outputTensor.get(), inputDataSize);
            }
        }
        else if (outputTensor->GetTensorDataType() == MLOperatorTensorDataType::Double &&
                 inputTensor->GetTensorDataType() == MLOperatorTensorDataType::Double)
        {
            // For cpu data
            if (outputTensor->IsCpuData() && inputTensor->IsCpuData())
            {
                ComputeInternal<double>(inputTensor.get(), outputTensor.get(), inputDataSize);
            }
        }


        return S_OK;
    }

    template <typename T, typename U = T>
    void ComputeInternal(IMLOperatorTensor* pInputTensor, IMLOperatorTensor* pOutputTensor, uint32_t size)
    {
        auto inputData = static_cast<T*>(pInputTensor->GetData());
        auto outputData = static_cast<U*>(pOutputTensor->GetData());

        for (uint32_t i = 0; i < size; i++)
        {
            outputData[i] = static_cast<U>(std::max<T>(0, inputData[i]));
        }
    }
};

struct ReluOperatorFactory : winrt::implements<ReluOperatorFactory, IMLOperatorKernelFactory>
{
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        IMLOperatorKernel** kernel)
    {
        ORT_UNUSED_PARAMETER(context);
        auto reluOperator = winrt::make<ReluOperator>();
        reluOperator.copy_to(kernel);
        return S_OK;
    }

    static MLOperatorEdgeDescription CreateEdgeDescriptor(MLOperatorEdgeType type, MLOperatorTensorDataType dataType)
    {
        ORT_UNUSED_PARAMETER(type);
        MLOperatorEdgeDescription desc;
        desc.edgeType = MLOperatorEdgeType::Tensor;
        desc.tensorDataType = dataType;
        return desc;
    }

    static void RegisterReluKernel(winrt::com_ptr<IMLOperatorRegistry> registry)
    {
        MLOperatorKernelDescription kernelDescription;
        kernelDescription.domain = "";
        kernelDescription.name = "Relu";
        kernelDescription.minimumOperatorSetVersion = 1;
        kernelDescription.executionType = MLOperatorExecutionType::Cpu;

        MLOperatorEdgeTypeConstrant typeConstraint;
        typeConstraint.typeLabel = "T";
        std::vector<MLOperatorEdgeDescription> allowedEdges
        {
            CreateEdgeDescriptor(MLOperatorEdgeType::Tensor, MLOperatorTensorDataType::Double),
            CreateEdgeDescriptor(MLOperatorEdgeType::Tensor, MLOperatorTensorDataType::Float),
            CreateEdgeDescriptor(MLOperatorEdgeType::Tensor, MLOperatorTensorDataType::Float16)
        };
        typeConstraint.allowedTypes = allowedEdges.data();
        typeConstraint.allowedTypeCount = static_cast<uint32_t>(allowedEdges.size());

        std::vector<MLOperatorEdgeTypeConstrant> typeConstraints{ typeConstraint };
        kernelDescription.typeConstraints = typeConstraints.data();
        kernelDescription.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        kernelDescription.defaultAttributes = nullptr;
        kernelDescription.defaultAttributeCount = 0;
        kernelDescription.options = MLOperatorKernelOptions::None;
        kernelDescription.executionOptions = 0;

        auto factory = winrt::make<ReluOperatorFactory>();
        auto shareInferrer = winrt::make<ReluShapeInferrer>();

        registry->RegisterOperatorKernel(
            &kernelDescription,
            factory.get(),
            shareInferrer.get()
        );
    }
};
