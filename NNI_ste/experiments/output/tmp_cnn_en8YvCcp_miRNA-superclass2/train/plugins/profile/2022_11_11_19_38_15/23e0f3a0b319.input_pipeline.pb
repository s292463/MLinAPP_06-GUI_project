	��ל@��ל@!��ל@	ox���@ox���@!ox���@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��ל@H�c�C��?1Q.�_x��?A����W�?ILݕ]0X@Yx$(~��?rEagerKernelExecute 0*	���w��@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����?!�`���<I@)Ǆ�K���?1X���#H@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map������?!`rE��F@)�#�&ݖ�?1�p�5�D@:Preprocessing2F
Iterator::Model�rf�B�?!9�މ]�@)3���/�?1��E�3@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat��+d��?!�`e6.@)�I|���?1iBt&� @:Preprocessing2U
Iterator::Model::ParallelMapV2�խ��ޗ?!-�1�v]�?)�խ��ޗ?1-�1�v]�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'������?!'�a���?)E�A��?1�JC����?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�r�4��?!B��l��?)��r��?1`��\��?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�<Fy��?!0y_����?)�<Fy��?10y_����?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�x�ߢ��?!����J@)�:�f�?1�-wd�y�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�Hh˹w?!I��9���?)�Hh˹w?1I��9���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceb�� ��t?!�<	�e8�?)b�� ��t?1�<	�e8�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range#�ng_yp?!��^����?)#�ng_yp?1��^����?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate����	�?!>��\��?)2�w�f?1�($�|�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorHP�s�R?!�屯�?)HP�s�R?1�屯�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 20.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�59.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ox���@I���$T@Qߺ�4�t.@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	H�c�C��?H�c�C��?!H�c�C��?      ��!       "	Q.�_x��?Q.�_x��?!Q.�_x��?*      ��!       2	����W�?����W�?!����W�?:	Lݕ]0X@Lݕ]0X@!Lݕ]0X@B      ��!       J	x$(~��?x$(~��?!x$(~��?R      ��!       Z	x$(~��?x$(~��?!x$(~��?b      ��!       JGPUYox���@b q���$T@yߺ�4�t.@