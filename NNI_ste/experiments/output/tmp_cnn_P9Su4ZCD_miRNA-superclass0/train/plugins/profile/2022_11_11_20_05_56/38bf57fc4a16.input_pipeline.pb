	UO?}? @UO?}? @!UO?}? @	??֖?$@??֖?$@!??֖?$@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLUO?}? @??(&o ??1???EҎ@A??x?'??I???????Y??q?d???rEagerKernelExecute 0*		?Zdd@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????O??!YE'?=?E@)??Z&????1F?}??.D@:Preprocessing2F
Iterator::Modeljj?Z_??!??(0?C@))????B??1l!???!8@:Preprocessing2U
Iterator::Model::ParallelMapV2??Pn????!	?%?"?-@)??Pn????1	?%?"?-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL?uT5??!Z֍$@)#??]???1_?z?Ί@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???M???!H?h??N@)M?]~??1???̤@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???ٕv?!??A͒?
@)???ٕv?1??A͒?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????5>??!V??Ss?F@)?o????m?1?7*V?@:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorpA?,_g?!????]???)pA?,_g?1????]???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice]?`7l[d?!??G|(???)]?`7l[d?1??G|(???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?21.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??֖?$@I)?ƙ??C@Q٬????L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??(&o ????(&o ??!??(&o ??      ??!       "	???EҎ@???EҎ@!???EҎ@*      ??!       2	??x?'????x?'??!??x?'??:	??????????????!???????B      ??!       J	??q?d?????q?d???!??q?d???R      ??!       Z	??q?d?????q?d???!??q?d???b      ??!       JGPUY??֖?$@b q)?ƙ??C@y٬????L@