	�9>Z��!@�9>Z��!@!�9>Z��!@	����M�?����M�?!����M�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�9>Z��!@A+0du+@1e��2��@A��R{m�?I��7i�@Y�E}�;l�?rEagerKernelExecute 0*	�l���=b@2F
Iterator::Model5ӽN�˲?!-|�d(I@){Cr2�?1\����@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����?!b���/=@)�Y�X�?1�f��88@:Preprocessing2U
Iterator::Model::ParallelMapV2�Vд�ʘ?!�CXW�0@)�Vд�ʘ?1�CXW�0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�ip[[�?!���C�L @)�ip[[�?1���C�L @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateτ&�%�?!ƊI	�M,@)g�ܶ�?1��]��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�8�Վ�|?!�׈T@)�8�Վ�|?1�׈T@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���}���?!���q��H@)줾,��|?1Q��iK@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapV+~���?!c=0:Y�/@)v�r��c?1�5�م�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 24.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�49.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����M�?I���]�R@Q��]��8@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	A+0du+@A+0du+@!A+0du+@      ��!       "	e��2��@e��2��@!e��2��@*      ��!       2	��R{m�?��R{m�?!��R{m�?:	��7i�@��7i�@!��7i�@B      ��!       J	�E}�;l�?�E}�;l�?!�E}�;l�?R      ��!       Z	�E}�;l�?�E}�;l�?!�E}�;l�?b      ��!       JGPUY����M�?b q���]�R@y��]��8@