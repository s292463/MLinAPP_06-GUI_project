	ݱ?&?@ݱ?&?@!ݱ?&?@	?y'˾
@?y'˾
@!?y'˾
@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLݱ?&?@ʉvR???1??~?? ??A??vhX???In??)?@Y2?m??f??rEagerKernelExecute 0*	]???(?b@2F
Iterator::Modelɐc???!wހ??G@)V,~SX???1????@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS?r/0+??!|̄)Y:@)??u?X??1ڧ	B[5@:Preprocessing2U
Iterator::Model::ParallelMapV2??F????!???
x'.@)??F????1???
x'.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenates֧?ŝ?!10dr3@)????ӓ?1\????)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceq?J[\???!?g?KS?@)q?J[\???1?g?KS?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??J
,??!?!?EZJ@)??hUM??1?}?L@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorz?΅?~?!?????@)z?΅?~?1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(?>???!%?3?a5@)䞮?Xlc?1az??_??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 26.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?44.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?y'˾
@I]?????Q@QG?h?*a9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ʉvR???ʉvR???!ʉvR???      ??!       "	??~?? ????~?? ??!??~?? ??*      ??!       2	??vhX?????vhX???!??vhX???:	n??)?@n??)?@!n??)?@B      ??!       J	2?m??f??2?m??f??!2?m??f??R      ??!       Z	2?m??f??2?m??f??!2?m??f??b      ??!       JGPUY?y'˾
@b q]?????Q@yG?h?*a9@