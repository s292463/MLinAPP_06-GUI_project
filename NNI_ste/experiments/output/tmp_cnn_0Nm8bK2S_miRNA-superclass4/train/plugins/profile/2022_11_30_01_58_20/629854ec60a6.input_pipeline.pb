	���;�9@���;�9@!���;�9@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:���;�9@�������?1�`7l[�7@I#ڎ����?rEagerKernelExecute 0*	�ʡE�t@2U
Iterator::Model::ParallelMapV2�L����?!��P�K@)�L����?1��P�K@:Preprocessing2F
Iterator::Model�/�'�?!�j� [�Q@) �g��?�?1��j�2�.@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��/��?!�x��h0@)�B,cC�?1� ӧ�L,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceY���RA�?!3��YW�@)Y���RA�?13��YW�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��0��?!�U���V=@)q㊋��?1�;aR��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate]T���?!'6$�n�"@):ZՒ��?14��P�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�1>�^�}?!��7�@)�1>�^�}?1��7�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap@��wԘ�?!�k3��0$@)��fHe?1X����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��}Н@Q��$�"W@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�������?�������?!�������?      ��!       "	�`7l[�7@�`7l[�7@!�`7l[�7@*      ��!       2      ��!       :	#ڎ����?#ڎ����?!#ڎ����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��}Н@y��$�"W@