	??w﨑)@??w﨑)@!??w﨑)@	????????????!??????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??w﨑)@稣?jd??11%??%@A?^?"??}?Io???I???Y@k~??E??rEagerKernelExecute 0*	?v???d@2F
Iterator::Modelqvk?ǳ?!'??8?AG@)1?Z{????1?O??b[?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatN+?@.q??!?<???	8@)?GQg?!??1pB#d%4@:Preprocessing2U
Iterator::Model::ParallelMapV2`?n?ƙ?!J??O.@)`?n?ƙ?1J??O.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateԞ?sb??!wȿ??1@)dt@????1?I?Be?(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??*????!lZ?b??8@)????Ӊ?1?GR??^@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?T???B??!B?4=@)?T???B??1B?4=@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipd???H???!?S?c?J@)???~?:??13???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????yz?!?h??4"@)????yz?1?h??4"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?13.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??????I,?a??/@Q.FCʝT@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	稣?jd??稣?jd??!稣?jd??      ??!       "	1%??%@1%??%@!1%??%@*      ??!       2	?^?"??}??^?"??}?!?^?"??}?:	o???I???o???I???!o???I???B      ??!       J	@k~??E??@k~??E??!@k~??E??R      ??!       Z	@k~??E??@k~??E??!@k~??E??b      ??!       JGPUY??????b q,?a??/@y.FCʝT@