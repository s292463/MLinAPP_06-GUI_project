	O=��V%@O=��V%@!O=��V%@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:O=��V%@}гY���?1YR�>�@I����@rEagerKernelExecute 0*	��|?5�u@2U
Iterator::Model::ParallelMapV2�+�V]��?!k�_��K@)�+�V]��?1k�_��K@:Preprocessing2F
Iterator::Model�6���?!2r�.R@)�>�-W?�?1�>��"1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatir1�q�?!���@0@)�-���?1��L�9�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice'��>��?!�z.@)'��>��?1�z.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�~�x���?!�L���@)AJ�i�?1��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�x��?!:7��G;@)��x��M�?1���Z��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensork*��.�~?!����M@)k*��.�~?1����M@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapڬ�\mŞ?!��U�n!@)r�#Df?1&K�N:�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�29.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�����?@Q��~SQ@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	}гY���?}гY���?!}гY���?      ��!       "	YR�>�@YR�>�@!YR�>�@*      ��!       2      ��!       :	����@����@!����@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�����?@y��~SQ@