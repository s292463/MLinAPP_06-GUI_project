	�T��@�T��@!�T��@	���@���@!���@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�T��@��0�*�?1���vh��?A�g�o}X�?I%̴��@YV]��?rEagerKernelExecute 0*	C�l��1c@2F
Iterator::Model� x|{װ?!�:��kE@)c �={�?1�I�C��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatl@��r��?!�Iۮ4=@)A�mߣ��?1�e7]�(8@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���W�<�?!: ,ߍ-@)���W�<�?1: ,ߍ-@:Preprocessing2U
Iterator::Model::ParallelMapV2�q��rg�?!q��~,@)�q��rg�?1q��~,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�n��?!j�c�!}4@)O�9����?1�ю���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipip[[x�?!`��]O�L@)
�\���?1�##�	�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorX9��v�?!��H��/@)X9��v�?1��H��/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�q��Q��?!�x�}�6@)-@�j�i?1�:��_n @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�44.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���@I|� �j�Q@Q�^�:@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��0�*�?��0�*�?!��0�*�?      ��!       "	���vh��?���vh��?!���vh��?*      ��!       2	�g�o}X�?�g�o}X�?!�g�o}X�?:	%̴��@%̴��@!%̴��@B      ��!       J	V]��?V]��?!V]��?R      ��!       Z	V]��?V]��?!V]��?b      ��!       JGPUY���@b q|� �j�Q@y�^�:@