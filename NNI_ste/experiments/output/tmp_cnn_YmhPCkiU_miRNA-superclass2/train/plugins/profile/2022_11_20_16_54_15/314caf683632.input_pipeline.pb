	~����@~����@!~����@		4��H@	4��H@!	4��H@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL~����@vT5A�}�?1>���d@A���0�?I�y��@YE�N����?rEagerKernelExecute 0*	��x�&�c@2F
Iterator::Model�vhX���?!�;xt%G@)��S �g�?1�O�m-)>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateq9^��I�?!.hjr�@?@)�^a����?10
lf==@:Preprocessing2U
Iterator::Model::ParallelMapV2��`�?!�䂻!0@)��`�?1�䂻!0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipY�.���?!�ć��J@),G�@��?1f��JZM"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�#Di�?!��Nw��&@)D���XP�?13��h�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�?mT�y?!�m��@)�?mT�y?1�m��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�D2�ت?!Ĩ:�@@)�g^��h?1����{��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��N]?!݃}���?)��N]?1݃}���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��|	\?!\h�^Z�?)��|	\?1\h�^Z�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 17.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�46.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9	4��H@IFl3S�O@Qz7��+�@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	vT5A�}�?vT5A�}�?!vT5A�}�?      ��!       "	>���d@>���d@!>���d@*      ��!       2	���0�?���0�?!���0�?:	�y��@�y��@!�y��@B      ��!       J	E�N����?E�N����?!E�N����?R      ��!       Z	E�N����?E�N����?!E�N����?b      ��!       JGPUY	4��H@b qFl3S�O@yz7��+�@@