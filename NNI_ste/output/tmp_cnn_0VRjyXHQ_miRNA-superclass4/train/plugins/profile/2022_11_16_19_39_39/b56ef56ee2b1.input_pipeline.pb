	�_!se�@�_!se�@!�_!se�@	\�PdD@\�PdD@!\�PdD@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�_!se�@�h㈵��?1�d��?A/M��.�?I��^fX@YD�����?rEagerKernelExecute 0*	�A`���b@2F
Iterator::Model�>U�b�?!EHeoF@)� |�?1�X,�v�?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�1����?!��$��;@)����?1h|�gz?7@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicezq�ҕ?!�Є�*),@)zq�ҕ?1�Є�*),@:Preprocessing2U
Iterator::Model::ParallelMapV2��6�ُ�?!���\`�*@)��6�ُ�?1���\`�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����[�?!S���9�4@)c+hZbe�?1��T��R@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipKO�\�?!�����K@)����W~?1�i��{�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn�8)�{|?!�-�Ja@)n�8)�{|?1�-�Ja@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��AA)Z�?!!�!�d6@)��f�|e?1�,Ƴ6��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 22.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�41.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9\�PdD@IJ��͡%P@Q���_?@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�h㈵��?�h㈵��?!�h㈵��?      ��!       "	�d��?�d��?!�d��?*      ��!       2	/M��.�?/M��.�?!/M��.�?:	��^fX@��^fX@!��^fX@B      ��!       J	D�����?D�����?!D�����?R      ��!       Z	D�����?D�����?!D�����?b      ��!       JGPUY\�PdD@b qJ��͡%P@y���_?@