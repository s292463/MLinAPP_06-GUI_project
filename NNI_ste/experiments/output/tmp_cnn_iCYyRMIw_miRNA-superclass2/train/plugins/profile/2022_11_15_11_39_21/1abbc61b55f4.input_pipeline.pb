	??@??@!??@	Ku??ǽ@Ku??ǽ@!Ku??ǽ@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??@x??1!???1???n' @A[??d9??IT8?T*@Y?^)????rEagerKernelExecute 0*	o????b@2F
Iterator::ModelobHN&n??!&?<y?F@)-???a??1?|?6@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatנ/?????!?4?????@)?7ӅX??1T3?Ѕ?;@:Preprocessing2U
Iterator::Model::ParallelMapV2c??Ց??!???o?M)@)c??Ց??1???o?M)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicemFA????!?????@)mFA????1?????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateH?m??ʖ?!?(jU?w-@)#-??#???1?????;@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipk???t=??!?E?ÆvK@)?B:<????1:??U?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorcG?P?{?!??"|@)cG?P?{?1??"|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?f׽???!??x?)?0@)?N^?e?1??;?c??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?38.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t21.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9Ku??ǽ@I5ҏ?FN@Q!?u?7"@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	x??1!???x??1!???!x??1!???      ??!       "	???n' @???n' @!???n' @*      ??!       2	[??d9??[??d9??![??d9??:	T8?T*@T8?T*@!T8?T*@B      ??!       J	?^)?????^)????!?^)????R      ??!       Z	?^)?????^)????!?^)????b      ??!       JGPUYKu??ǽ@b q5ҏ?FN@y!?u?7"@@