	4���vWr@4���vWr@!4���vWr@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC4���vWr@<��Ӹ7�?1��1p@Aϼv�1�?I����s@@rEagerKernelExecute 0*	��Q�!d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateQ1�߄B�?!>\eЊ�C@)?�'i�?1�T8�#qB@:Preprocessing2F
Iterator::Model/j�� ߱?!���vӬE@)�m3⑨?1��L��=@:Preprocessing2U
Iterator::Model::ParallelMapV2��v�>X�?!��lA�+@)��v�>X�?1��lA�+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��Y��B�?!��4x�\'@)�o`r�Ȋ?1�y)> @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipN~�N�Z�?!J1�,SL@)�t��{?1<FąMi@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���pzw?!Ew-��y@)���pzw?1Ew-��y@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��A��c?!��Я�?)��A��c?1��Я�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�O���ذ?!Y�k��nD@)��C���b?1N��@���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice>?�m\?!)d��=�?)>?�m\?1)d��=�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�11.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�*k�r'@Q���V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	<��Ӹ7�?<��Ӹ7�?!<��Ӹ7�?      ��!       "	��1p@��1p@!��1p@*      ��!       2	ϼv�1�?ϼv�1�?!ϼv�1�?:	����s@@����s@@!����s@@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�*k�r'@y���V@