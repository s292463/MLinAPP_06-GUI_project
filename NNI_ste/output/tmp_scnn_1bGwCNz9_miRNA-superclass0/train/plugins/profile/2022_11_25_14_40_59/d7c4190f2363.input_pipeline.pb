	r���=@r���=@!r���=@	C3%_;@C3%_;@!C3%_;@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0r���=@ByGs�?1�]J]22@I֋��h�#@YYO���j�?r0*	��Q���@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/�ͮ�@!P��U�L@)P�s'�@1Q����L@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map? �M�\@!���]��D@)�Z{��@1��K�zD@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat.Ui�k|�?!���[�p�?)Z��8��?1\�X3��?:Preprocessing2F
Iterator::Model{נ/���?!�@t�� �?)a��w}�?1²�r!�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZc�	���?!��L��O�?)]7��VB�?1<[���?:Preprocessing2U
Iterator::Model::ParallelMapV2��F���?! �[c��?)��F���?1 �[c��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��m��@!a�f�Y�L@)�c���Ȕ?1�!� �d�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlicea7l[�ِ?!"!ˣ�?)a7l[�ِ?1"!ˣ�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�܁:�?!����-+�?)�܁:�?1����-+�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���4)}?!�q���7�?)���4)}?1�q���7�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�&�|�w?!Z�|�ݰ?)�&�|�w?1Z�|�ݰ?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�
E��S`?!Ɓi��ߗ?)�
E��S`?1Ɓi��ߗ?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�32.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9C3%_;@I�Ň!��@@Qϓ��4N@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ByGs�?ByGs�?!ByGs�?      ��!       "	�]J]22@�]J]22@!�]J]22@*      ��!       2      ��!       :	֋��h�#@֋��h�#@!֋��h�#@B      ��!       J	YO���j�?YO���j�?!YO���j�?R      ��!       Z	YO���j�?YO���j�?!YO���j�?b      ��!       JGPUYC3%_;@b q�Ň!��@@yϓ��4N@