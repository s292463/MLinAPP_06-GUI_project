??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??

~
Conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:[* 
shared_nameConv1D_1/kernel
w
#Conv1D_1/kernel/Read/ReadVariableOpReadVariableOpConv1D_1/kernel*"
_output_shapes
:[*
dtype0
r
Conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*
shared_nameConv1D_1/bias
k
!Conv1D_1/bias/Read/ReadVariableOpReadVariableOpConv1D_1/bias*
_output_shapes
:[*
dtype0
~
Conv1D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:[J* 
shared_nameConv1D_2/kernel
w
#Conv1D_2/kernel/Read/ReadVariableOpReadVariableOpConv1D_2/kernel*"
_output_shapes
:[J*
dtype0
r
Conv1D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*
shared_nameConv1D_2/bias
k
!Conv1D_2/bias/Read/ReadVariableOpReadVariableOpConv1D_2/bias*
_output_shapes
:J*
dtype0
~
Conv1D_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:JL* 
shared_nameConv1D_3/kernel
w
#Conv1D_3/kernel/Read/ReadVariableOpReadVariableOpConv1D_3/kernel*"
_output_shapes
:JL*
dtype0
r
Conv1D_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:L*
shared_nameConv1D_3/bias
k
!Conv1D_3/bias/Read/ReadVariableOpReadVariableOpConv1D_3/bias*
_output_shapes
:L*
dtype0
~
Conv1D_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:LJ* 
shared_nameConv1D_4/kernel
w
#Conv1D_4/kernel/Read/ReadVariableOpReadVariableOpConv1D_4/kernel*"
_output_shapes
:LJ*
dtype0
r
Conv1D_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*
shared_nameConv1D_4/bias
k
!Conv1D_4/bias/Read/ReadVariableOpReadVariableOpConv1D_4/bias*
_output_shapes
:J*
dtype0
z
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameDense_1/kernel
s
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel* 
_output_shapes
:
??*
dtype0
q
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_1/bias
j
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes	
:?*
dtype0
y
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameDense_2/kernel
r
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel*
_output_shapes
:	?*
dtype0
p
Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense_2/bias
i
 Dense_2/bias/Read/ReadVariableOpReadVariableOpDense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/Conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*'
shared_nameAdam/Conv1D_1/kernel/m
?
*Adam/Conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/m*"
_output_shapes
:[*
dtype0
?
Adam/Conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*%
shared_nameAdam/Conv1D_1/bias/m
y
(Adam/Conv1D_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/m*
_output_shapes
:[*
dtype0
?
Adam/Conv1D_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:[J*'
shared_nameAdam/Conv1D_2/kernel/m
?
*Adam/Conv1D_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/m*"
_output_shapes
:[J*
dtype0
?
Adam/Conv1D_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*%
shared_nameAdam/Conv1D_2/bias/m
y
(Adam/Conv1D_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/m*
_output_shapes
:J*
dtype0
?
Adam/Conv1D_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:JL*'
shared_nameAdam/Conv1D_3/kernel/m
?
*Adam/Conv1D_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/kernel/m*"
_output_shapes
:JL*
dtype0
?
Adam/Conv1D_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:L*%
shared_nameAdam/Conv1D_3/bias/m
y
(Adam/Conv1D_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/bias/m*
_output_shapes
:L*
dtype0
?
Adam/Conv1D_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:LJ*'
shared_nameAdam/Conv1D_4/kernel/m
?
*Adam/Conv1D_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/kernel/m*"
_output_shapes
:LJ*
dtype0
?
Adam/Conv1D_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*%
shared_nameAdam/Conv1D_4/bias/m
y
(Adam/Conv1D_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/bias/m*
_output_shapes
:J*
dtype0
?
Adam/Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/Dense_1/kernel/m
?
)Adam/Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/Dense_1/bias/m
x
'Adam/Dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/Dense_2/kernel/m
?
)Adam/Dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/Dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/Dense_2/bias/m
w
'Adam/Dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/Conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*'
shared_nameAdam/Conv1D_1/kernel/v
?
*Adam/Conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/v*"
_output_shapes
:[*
dtype0
?
Adam/Conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*%
shared_nameAdam/Conv1D_1/bias/v
y
(Adam/Conv1D_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/v*
_output_shapes
:[*
dtype0
?
Adam/Conv1D_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:[J*'
shared_nameAdam/Conv1D_2/kernel/v
?
*Adam/Conv1D_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/v*"
_output_shapes
:[J*
dtype0
?
Adam/Conv1D_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*%
shared_nameAdam/Conv1D_2/bias/v
y
(Adam/Conv1D_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/v*
_output_shapes
:J*
dtype0
?
Adam/Conv1D_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:JL*'
shared_nameAdam/Conv1D_3/kernel/v
?
*Adam/Conv1D_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/kernel/v*"
_output_shapes
:JL*
dtype0
?
Adam/Conv1D_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:L*%
shared_nameAdam/Conv1D_3/bias/v
y
(Adam/Conv1D_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/bias/v*
_output_shapes
:L*
dtype0
?
Adam/Conv1D_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:LJ*'
shared_nameAdam/Conv1D_4/kernel/v
?
*Adam/Conv1D_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/kernel/v*"
_output_shapes
:LJ*
dtype0
?
Adam/Conv1D_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*%
shared_nameAdam/Conv1D_4/bias/v
y
(Adam/Conv1D_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/bias/v*
_output_shapes
:J*
dtype0
?
Adam/Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/Dense_1/kernel/v
?
)Adam/Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/Dense_1/bias/v
x
'Adam/Dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/Dense_2/kernel/v
?
)Adam/Dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/Dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/Dense_2/bias/v
w
'Adam/Dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?H
value?HB?H B?H
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
R
6	variables
7regularization_losses
8trainable_variables
9	keras_api
h

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
h

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
?
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_ratem?m?m?m?&m?'m?0m?1m?:m?;m?@m?Am?v?v?v?v?&v?'v?0v?1v?:v?;v?@v?Av?
V
0
1
2
3
&4
'5
06
17
:8
;9
@10
A11
 
V
0
1
2
3
&4
'5
06
17
:8
;9
@10
A11
?
	variables
Klayer_regularization_losses
regularization_losses
Lmetrics
trainable_variables
Mnon_trainable_variables
Nlayer_metrics

Olayers
 
[Y
VARIABLE_VALUEConv1D_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Player_regularization_losses
	variables
regularization_losses
Qmetrics
trainable_variables
Rnon_trainable_variables
Slayer_metrics

Tlayers
 
 
 
?
Ulayer_regularization_losses
	variables
regularization_losses
Vmetrics
trainable_variables
Wnon_trainable_variables
Xlayer_metrics

Ylayers
[Y
VARIABLE_VALUEConv1D_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Zlayer_regularization_losses
	variables
regularization_losses
[metrics
 trainable_variables
\non_trainable_variables
]layer_metrics

^layers
 
 
 
?
_layer_regularization_losses
"	variables
#regularization_losses
`metrics
$trainable_variables
anon_trainable_variables
blayer_metrics

clayers
[Y
VARIABLE_VALUEConv1D_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
?
dlayer_regularization_losses
(	variables
)regularization_losses
emetrics
*trainable_variables
fnon_trainable_variables
glayer_metrics

hlayers
 
 
 
?
ilayer_regularization_losses
,	variables
-regularization_losses
jmetrics
.trainable_variables
knon_trainable_variables
llayer_metrics

mlayers
[Y
VARIABLE_VALUEConv1D_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
?
nlayer_regularization_losses
2	variables
3regularization_losses
ometrics
4trainable_variables
pnon_trainable_variables
qlayer_metrics

rlayers
 
 
 
?
slayer_regularization_losses
6	variables
7regularization_losses
tmetrics
8trainable_variables
unon_trainable_variables
vlayer_metrics

wlayers
ZX
VARIABLE_VALUEDense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
?
xlayer_regularization_losses
<	variables
=regularization_losses
ymetrics
>trainable_variables
znon_trainable_variables
{layer_metrics

|layers
ZX
VARIABLE_VALUEDense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
?
}layer_regularization_losses
B	variables
Cregularization_losses
~metrics
Dtrainable_variables
non_trainable_variables
?layer_metrics
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/Conv1D_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Conv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasConv1D_3/kernelConv1D_3/biasConv1D_4/kernelConv1D_4/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_12678
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv1D_1/kernel/Read/ReadVariableOp!Conv1D_1/bias/Read/ReadVariableOp#Conv1D_2/kernel/Read/ReadVariableOp!Conv1D_2/bias/Read/ReadVariableOp#Conv1D_3/kernel/Read/ReadVariableOp!Conv1D_3/bias/Read/ReadVariableOp#Conv1D_4/kernel/Read/ReadVariableOp!Conv1D_4/bias/Read/ReadVariableOp"Dense_1/kernel/Read/ReadVariableOp Dense_1/bias/Read/ReadVariableOp"Dense_2/kernel/Read/ReadVariableOp Dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Conv1D_1/kernel/m/Read/ReadVariableOp(Adam/Conv1D_1/bias/m/Read/ReadVariableOp*Adam/Conv1D_2/kernel/m/Read/ReadVariableOp(Adam/Conv1D_2/bias/m/Read/ReadVariableOp*Adam/Conv1D_3/kernel/m/Read/ReadVariableOp(Adam/Conv1D_3/bias/m/Read/ReadVariableOp*Adam/Conv1D_4/kernel/m/Read/ReadVariableOp(Adam/Conv1D_4/bias/m/Read/ReadVariableOp)Adam/Dense_1/kernel/m/Read/ReadVariableOp'Adam/Dense_1/bias/m/Read/ReadVariableOp)Adam/Dense_2/kernel/m/Read/ReadVariableOp'Adam/Dense_2/bias/m/Read/ReadVariableOp*Adam/Conv1D_1/kernel/v/Read/ReadVariableOp(Adam/Conv1D_1/bias/v/Read/ReadVariableOp*Adam/Conv1D_2/kernel/v/Read/ReadVariableOp(Adam/Conv1D_2/bias/v/Read/ReadVariableOp*Adam/Conv1D_3/kernel/v/Read/ReadVariableOp(Adam/Conv1D_3/bias/v/Read/ReadVariableOp*Adam/Conv1D_4/kernel/v/Read/ReadVariableOp(Adam/Conv1D_4/bias/v/Read/ReadVariableOp)Adam/Dense_1/kernel/v/Read/ReadVariableOp'Adam/Dense_1/bias/v/Read/ReadVariableOp)Adam/Dense_2/kernel/v/Read/ReadVariableOp'Adam/Dense_2/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_13202
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasConv1D_3/kernelConv1D_3/biasConv1D_4/kernelConv1D_4/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv1D_1/kernel/mAdam/Conv1D_1/bias/mAdam/Conv1D_2/kernel/mAdam/Conv1D_2/bias/mAdam/Conv1D_3/kernel/mAdam/Conv1D_3/bias/mAdam/Conv1D_4/kernel/mAdam/Conv1D_4/bias/mAdam/Dense_1/kernel/mAdam/Dense_1/bias/mAdam/Dense_2/kernel/mAdam/Dense_2/bias/mAdam/Conv1D_1/kernel/vAdam/Conv1D_1/bias/vAdam/Conv1D_2/kernel/vAdam/Conv1D_2/bias/vAdam/Conv1D_3/kernel/vAdam/Conv1D_3/bias/vAdam/Conv1D_4/kernel/vAdam/Conv1D_4/bias/vAdam/Dense_1/kernel/vAdam/Dense_1/bias/vAdam/Dense_2/kernel/vAdam/Dense_2/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_13347??
?m
?

@__inference_model_layer_call_and_return_conditional_losses_12815

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:[6
(conv1d_1_biasadd_readvariableop_resource:[J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:[J6
(conv1d_2_biasadd_readvariableop_resource:JJ
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:JL6
(conv1d_3_biasadd_readvariableop_resource:LJ
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:LJ6
(conv1d_4_biasadd_readvariableop_resource:J:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:
identity??Conv1D_1/BiasAdd/ReadVariableOp?+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_2/BiasAdd/ReadVariableOp?+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_3/BiasAdd/ReadVariableOp?+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_4/BiasAdd/ReadVariableOp?+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_1/conv1d/ExpandDims/dim?
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_1/conv1d/ExpandDims?
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dim?
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[2
Conv1D_1/conv1d/ExpandDims_1?
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????[*
paddingVALID*
strides
2
Conv1D_1/conv1d?
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????[*
squeeze_dims

?????????2
Conv1D_1/conv1d/Squeeze?
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:[*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOp?
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????[2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????[2
Conv1D_1/Relu?
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dim?
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????[2
MaxPooling1D_1/ExpandDims?
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:??????????[*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPool?
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:??????????[*
squeeze_dims
2
MaxPooling1D_1/Squeeze?
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_2/conv1d/ExpandDims/dim?
Conv1D_2/conv1d/ExpandDims
ExpandDimsMaxPooling1D_1/Squeeze:output:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????[2
Conv1D_2/conv1d/ExpandDims?
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[J*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dim?
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[J2
Conv1D_2/conv1d/ExpandDims_1?
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????J*
paddingVALID*
strides
2
Conv1D_2/conv1d?
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*,
_output_shapes
:??????????J*
squeeze_dims

?????????2
Conv1D_2/conv1d/Squeeze?
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOp?
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????J2
Conv1D_2/BiasAddx
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????J2
Conv1D_2/Relu?
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dim?
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????J2
MaxPooling1D_2/ExpandDims?
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????IJ*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPool?
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????IJ*
squeeze_dims
2
MaxPooling1D_2/Squeeze?
Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_3/conv1d/ExpandDims/dim?
Conv1D_3/conv1d/ExpandDims
ExpandDimsMaxPooling1D_2/Squeeze:output:0'Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????IJ2
Conv1D_3/conv1d/ExpandDims?
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:JL*
dtype02-
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_3/conv1d/ExpandDims_1/dim?
Conv1D_3/conv1d/ExpandDims_1
ExpandDims3Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:JL2
Conv1D_3/conv1d/ExpandDims_1?
Conv1D_3/conv1dConv2D#Conv1D_3/conv1d/ExpandDims:output:0%Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????FL*
paddingVALID*
strides
2
Conv1D_3/conv1d?
Conv1D_3/conv1d/SqueezeSqueezeConv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????FL*
squeeze_dims

?????????2
Conv1D_3/conv1d/Squeeze?
Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:L*
dtype02!
Conv1D_3/BiasAdd/ReadVariableOp?
Conv1D_3/BiasAddBiasAdd Conv1D_3/conv1d/Squeeze:output:0'Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????FL2
Conv1D_3/BiasAddw
Conv1D_3/ReluReluConv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????FL2
Conv1D_3/Relu?
MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_3/ExpandDims/dim?
MaxPooling1D_3/ExpandDims
ExpandDimsConv1D_3/Relu:activations:0&MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????FL2
MaxPooling1D_3/ExpandDims?
MaxPooling1D_3/MaxPoolMaxPool"MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????L*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_3/MaxPool?
MaxPooling1D_3/SqueezeSqueezeMaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????L*
squeeze_dims
2
MaxPooling1D_3/Squeeze?
Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_4/conv1d/ExpandDims/dim?
Conv1D_4/conv1d/ExpandDims
ExpandDimsMaxPooling1D_3/Squeeze:output:0'Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????L2
Conv1D_4/conv1d/ExpandDims?
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:LJ*
dtype02-
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_4/conv1d/ExpandDims_1/dim?
Conv1D_4/conv1d/ExpandDims_1
ExpandDims3Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:LJ2
Conv1D_4/conv1d/ExpandDims_1?
Conv1D_4/conv1dConv2D#Conv1D_4/conv1d/ExpandDims:output:0%Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????J*
paddingVALID*
strides
2
Conv1D_4/conv1d?
Conv1D_4/conv1d/SqueezeSqueezeConv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????J*
squeeze_dims

?????????2
Conv1D_4/conv1d/Squeeze?
Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02!
Conv1D_4/BiasAdd/ReadVariableOp?
Conv1D_4/BiasAddBiasAdd Conv1D_4/conv1d/Squeeze:output:0'Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????J2
Conv1D_4/BiasAddw
Conv1D_4/ReluReluConv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????J2
Conv1D_4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????V  2
flatten/Const?
flatten/ReshapeReshapeConv1D_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMulflatten/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/BiasAddq
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_1/Relu?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Dense_2/BiasAdd?
IdentityIdentityDense_2/BiasAdd:output:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_3/BiasAdd/ReadVariableOp,^Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_4/BiasAdd/ReadVariableOp,^Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2B
Conv1D_1/BiasAdd/ReadVariableOpConv1D_1/BiasAdd/ReadVariableOp2Z
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_2/BiasAdd/ReadVariableOpConv1D_2/BiasAdd/ReadVariableOp2Z
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_3/BiasAdd/ReadVariableOpConv1D_3/BiasAdd/ReadVariableOp2Z
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_4/BiasAdd/ReadVariableOpConv1D_4/BiasAdd/ReadVariableOp2Z
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Dense_1_layer_call_fn_13014

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_123242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Dense_2_layer_call_fn_13034

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_123402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_12999

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_123112
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J:S O
+
_output_shapes
:?????????J
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_12919

inputsA
+conv1d_expanddims_1_readvariableop_resource:[-
biasadd_readvariableop_resource:[
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????[*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????[*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:[*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????[2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????[2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????[2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_12299

inputsA
+conv1d_expanddims_1_readvariableop_resource:LJ-
biasadd_readvariableop_resource:J
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????L2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:LJ*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:LJ2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????J*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????J*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????J2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????J2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????J2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????L: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????L
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_12565
input_1
unknown:[
	unknown_0:[
	unknown_1:[J
	unknown_2:J
	unknown_3:JL
	unknown_4:L
	unknown_5:LJ
	unknown_6:J
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_125092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
e
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_12201

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_12276

inputsA
+conv1d_expanddims_1_readvariableop_resource:JL-
biasadd_readvariableop_resource:L
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????IJ2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:JL*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:JL2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????FL*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????FL*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:L*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????FL2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????FL2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????FL2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????IJ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????IJ
 
_user_specified_nameinputs
?-
?
@__inference_model_layer_call_and_return_conditional_losses_12641
input_1$
conv1d_1_12606:[
conv1d_1_12608:[$
conv1d_2_12612:[J
conv1d_2_12614:J$
conv1d_3_12618:JL
conv1d_3_12620:L$
conv1d_4_12624:LJ
conv1d_4_12626:J!
dense_1_12630:
??
dense_1_12632:	? 
dense_2_12635:	?
dense_2_12637:
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_12606conv1d_1_12608*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_122302"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_121712 
MaxPooling1D_1/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_12612conv1d_2_12614*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_122532"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????IJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_121862 
MaxPooling1D_2/PartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0conv1d_3_12618conv1d_3_12620*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????FL*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_122762"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????L* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_122012 
MaxPooling1D_3/PartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0conv1d_4_12624conv1d_4_12626*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_122992"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_123112
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_12630dense_1_12632*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_123242!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_12635dense_2_12637*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_123402!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_13005

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????V  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J:S O
+
_output_shapes
:?????????J
 
_user_specified_nameinputs
?\
?
__inference__traced_save_13202
file_prefix.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableop5
1savev2_adam_conv1d_4_kernel_m_read_readvariableop3
/savev2_adam_conv1d_4_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableop5
1savev2_adam_conv1d_4_kernel_v_read_readvariableop3
/savev2_adam_conv1d_4_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop1savev2_adam_conv1d_4_kernel_m_read_readvariableop/savev2_adam_conv1d_4_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop1savev2_adam_conv1d_4_kernel_v_read_readvariableop/savev2_adam_conv1d_4_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :[:[:[J:J:JL:L:LJ:J:
??:?:	?:: : : : : : : : : :[:[:[J:J:JL:L:LJ:J:
??:?:	?::[:[:[J:J:JL:L:LJ:J:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:[: 

_output_shapes
:[:($
"
_output_shapes
:[J: 

_output_shapes
:J:($
"
_output_shapes
:JL: 

_output_shapes
:L:($
"
_output_shapes
:LJ: 

_output_shapes
:J:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:[: 

_output_shapes
:[:($
"
_output_shapes
:[J: 

_output_shapes
:J:($
"
_output_shapes
:JL: 

_output_shapes
:L:($
"
_output_shapes
:LJ: 

_output_shapes
:J:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::("$
"
_output_shapes
:[: #

_output_shapes
:[:($$
"
_output_shapes
:[J: %

_output_shapes
:J:(&$
"
_output_shapes
:JL: '

_output_shapes
:L:(($
"
_output_shapes
:LJ: )

_output_shapes
:J:&*"
 
_output_shapes
:
??:!+

_output_shapes	
:?:%,!

_output_shapes
:	?: -

_output_shapes
::.

_output_shapes
: 
?

?
B__inference_Dense_1_layer_call_and_return_conditional_losses_13025

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
@__inference_model_layer_call_and_return_conditional_losses_12603
input_1$
conv1d_1_12568:[
conv1d_1_12570:[$
conv1d_2_12574:[J
conv1d_2_12576:J$
conv1d_3_12580:JL
conv1d_3_12582:L$
conv1d_4_12586:LJ
conv1d_4_12588:J!
dense_1_12592:
??
dense_1_12594:	? 
dense_2_12597:	?
dense_2_12599:
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_12568conv1d_1_12570*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_122302"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_121712 
MaxPooling1D_1/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_12574conv1d_2_12576*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_122532"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????IJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_121862 
MaxPooling1D_2/PartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0conv1d_3_12580conv1d_3_12582*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????FL*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_122762"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????L* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_122012 
MaxPooling1D_3/PartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0conv1d_4_12586conv1d_4_12588*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_122992"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_123112
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_12592dense_1_12594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_123242!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_12597dense_2_12599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_123402!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_12994

inputsA
+conv1d_expanddims_1_readvariableop_resource:LJ-
biasadd_readvariableop_resource:J
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????L2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:LJ*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:LJ2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????J*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????J*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????J2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????J2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????J2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????L: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????L
 
_user_specified_nameinputs
?	
?
B__inference_Dense_2_layer_call_and_return_conditional_losses_12340

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_Dense_1_layer_call_and_return_conditional_losses_12324

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_12253

inputsA
+conv1d_expanddims_1_readvariableop_resource:[J-
biasadd_readvariableop_resource:J
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????[2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[J*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[J2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????J*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????J*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????J2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????J2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????J2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????[: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????[
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_12969

inputsA
+conv1d_expanddims_1_readvariableop_resource:JL-
biasadd_readvariableop_resource:L
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????IJ2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:JL*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:JL2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????FL*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????FL*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:L*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????FL2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????FL2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????FL2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????IJ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????IJ
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_4_layer_call_fn_12978

inputs
unknown:LJ
	unknown_0:J
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_122992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????J2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????L: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????L
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_2_layer_call_fn_12928

inputs
unknown:[J
	unknown_0:J
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_122532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????J2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????[: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????[
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_3_layer_call_fn_12207

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_122012
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
B__inference_Dense_2_layer_call_and_return_conditional_losses_13044

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_1_layer_call_fn_12177

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_121712
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_12311

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????V  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J:S O
+
_output_shapes
:?????????J
 
_user_specified_nameinputs
?m
?

@__inference_model_layer_call_and_return_conditional_losses_12894

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:[6
(conv1d_1_biasadd_readvariableop_resource:[J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:[J6
(conv1d_2_biasadd_readvariableop_resource:JJ
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:JL6
(conv1d_3_biasadd_readvariableop_resource:LJ
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:LJ6
(conv1d_4_biasadd_readvariableop_resource:J:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:
identity??Conv1D_1/BiasAdd/ReadVariableOp?+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_2/BiasAdd/ReadVariableOp?+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_3/BiasAdd/ReadVariableOp?+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_4/BiasAdd/ReadVariableOp?+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_1/conv1d/ExpandDims/dim?
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_1/conv1d/ExpandDims?
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dim?
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[2
Conv1D_1/conv1d/ExpandDims_1?
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????[*
paddingVALID*
strides
2
Conv1D_1/conv1d?
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????[*
squeeze_dims

?????????2
Conv1D_1/conv1d/Squeeze?
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:[*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOp?
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????[2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????[2
Conv1D_1/Relu?
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dim?
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????[2
MaxPooling1D_1/ExpandDims?
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:??????????[*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPool?
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:??????????[*
squeeze_dims
2
MaxPooling1D_1/Squeeze?
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_2/conv1d/ExpandDims/dim?
Conv1D_2/conv1d/ExpandDims
ExpandDimsMaxPooling1D_1/Squeeze:output:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????[2
Conv1D_2/conv1d/ExpandDims?
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[J*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dim?
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[J2
Conv1D_2/conv1d/ExpandDims_1?
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????J*
paddingVALID*
strides
2
Conv1D_2/conv1d?
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*,
_output_shapes
:??????????J*
squeeze_dims

?????????2
Conv1D_2/conv1d/Squeeze?
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOp?
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????J2
Conv1D_2/BiasAddx
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????J2
Conv1D_2/Relu?
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dim?
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????J2
MaxPooling1D_2/ExpandDims?
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????IJ*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPool?
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????IJ*
squeeze_dims
2
MaxPooling1D_2/Squeeze?
Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_3/conv1d/ExpandDims/dim?
Conv1D_3/conv1d/ExpandDims
ExpandDimsMaxPooling1D_2/Squeeze:output:0'Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????IJ2
Conv1D_3/conv1d/ExpandDims?
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:JL*
dtype02-
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_3/conv1d/ExpandDims_1/dim?
Conv1D_3/conv1d/ExpandDims_1
ExpandDims3Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:JL2
Conv1D_3/conv1d/ExpandDims_1?
Conv1D_3/conv1dConv2D#Conv1D_3/conv1d/ExpandDims:output:0%Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????FL*
paddingVALID*
strides
2
Conv1D_3/conv1d?
Conv1D_3/conv1d/SqueezeSqueezeConv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????FL*
squeeze_dims

?????????2
Conv1D_3/conv1d/Squeeze?
Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:L*
dtype02!
Conv1D_3/BiasAdd/ReadVariableOp?
Conv1D_3/BiasAddBiasAdd Conv1D_3/conv1d/Squeeze:output:0'Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????FL2
Conv1D_3/BiasAddw
Conv1D_3/ReluReluConv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????FL2
Conv1D_3/Relu?
MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_3/ExpandDims/dim?
MaxPooling1D_3/ExpandDims
ExpandDimsConv1D_3/Relu:activations:0&MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????FL2
MaxPooling1D_3/ExpandDims?
MaxPooling1D_3/MaxPoolMaxPool"MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????L*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_3/MaxPool?
MaxPooling1D_3/SqueezeSqueezeMaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????L*
squeeze_dims
2
MaxPooling1D_3/Squeeze?
Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_4/conv1d/ExpandDims/dim?
Conv1D_4/conv1d/ExpandDims
ExpandDimsMaxPooling1D_3/Squeeze:output:0'Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????L2
Conv1D_4/conv1d/ExpandDims?
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:LJ*
dtype02-
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_4/conv1d/ExpandDims_1/dim?
Conv1D_4/conv1d/ExpandDims_1
ExpandDims3Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:LJ2
Conv1D_4/conv1d/ExpandDims_1?
Conv1D_4/conv1dConv2D#Conv1D_4/conv1d/ExpandDims:output:0%Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????J*
paddingVALID*
strides
2
Conv1D_4/conv1d?
Conv1D_4/conv1d/SqueezeSqueezeConv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????J*
squeeze_dims

?????????2
Conv1D_4/conv1d/Squeeze?
Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02!
Conv1D_4/BiasAdd/ReadVariableOp?
Conv1D_4/BiasAddBiasAdd Conv1D_4/conv1d/Squeeze:output:0'Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????J2
Conv1D_4/BiasAddw
Conv1D_4/ReluReluConv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????J2
Conv1D_4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????V  2
flatten/Const?
flatten/ReshapeReshapeConv1D_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMulflatten/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/BiasAddq
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_1/Relu?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Dense_2/BiasAdd?
IdentityIdentityDense_2/BiasAdd:output:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_3/BiasAdd/ReadVariableOp,^Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_4/BiasAdd/ReadVariableOp,^Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2B
Conv1D_1/BiasAdd/ReadVariableOpConv1D_1/BiasAdd/ReadVariableOp2Z
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_2/BiasAdd/ReadVariableOpConv1D_2/BiasAdd/ReadVariableOp2Z
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_3/BiasAdd/ReadVariableOpConv1D_3/BiasAdd/ReadVariableOp2Z
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_4/BiasAdd/ReadVariableOpConv1D_4/BiasAdd/ReadVariableOp2Z
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_12707

inputs
unknown:[
	unknown_0:[
	unknown_1:[J
	unknown_2:J
	unknown_3:JL
	unknown_4:L
	unknown_5:LJ
	unknown_6:J
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_123472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_12374
input_1
unknown:[
	unknown_0:[
	unknown_1:[J
	unknown_2:J
	unknown_3:JL
	unknown_4:L
	unknown_5:LJ
	unknown_6:J
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_123472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?-
?
@__inference_model_layer_call_and_return_conditional_losses_12509

inputs$
conv1d_1_12474:[
conv1d_1_12476:[$
conv1d_2_12480:[J
conv1d_2_12482:J$
conv1d_3_12486:JL
conv1d_3_12488:L$
conv1d_4_12492:LJ
conv1d_4_12494:J!
dense_1_12498:
??
dense_1_12500:	? 
dense_2_12503:	?
dense_2_12505:
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_12474conv1d_1_12476*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_122302"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_121712 
MaxPooling1D_1/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_12480conv1d_2_12482*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_122532"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????IJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_121862 
MaxPooling1D_2/PartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0conv1d_3_12486conv1d_3_12488*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????FL*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_122762"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????L* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_122012 
MaxPooling1D_3/PartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0conv1d_4_12492conv1d_4_12494*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_122992"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_123112
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_12498dense_1_12500*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_123242!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_12503dense_2_12505*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_123402!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_12230

inputsA
+conv1d_expanddims_1_readvariableop_resource:[-
biasadd_readvariableop_resource:[
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????[*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????[*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:[*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????[2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????[2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????[2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_12944

inputsA
+conv1d_expanddims_1_readvariableop_resource:[J-
biasadd_readvariableop_resource:J
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????[2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[J*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[J2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????J*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????J*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????J2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????J2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????J2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????[: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????[
 
_user_specified_nameinputs
?-
?
@__inference_model_layer_call_and_return_conditional_losses_12347

inputs$
conv1d_1_12231:[
conv1d_1_12233:[$
conv1d_2_12254:[J
conv1d_2_12256:J$
conv1d_3_12277:JL
conv1d_3_12279:L$
conv1d_4_12300:LJ
conv1d_4_12302:J!
dense_1_12325:
??
dense_1_12327:	? 
dense_2_12341:	?
dense_2_12343:
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_12231conv1d_1_12233*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_122302"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_121712 
MaxPooling1D_1/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_12254conv1d_2_12256*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_122532"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????IJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_121862 
MaxPooling1D_2/PartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0conv1d_3_12277conv1d_3_12279*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????FL*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_122762"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????L* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_122012 
MaxPooling1D_3/PartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0conv1d_4_12300conv1d_4_12302*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_122992"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_123112
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_12325dense_1_12327*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_123242!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_12341dense_2_12343*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_123402!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_12186

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_2_layer_call_fn_12192

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_121862
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_1_layer_call_fn_12903

inputs
unknown:[
	unknown_0:[
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????[*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_122302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????[2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?y
?
 __inference__wrapped_model_12162
input_1P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:[<
.model_conv1d_1_biasadd_readvariableop_resource:[P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:[J<
.model_conv1d_2_biasadd_readvariableop_resource:JP
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:JL<
.model_conv1d_3_biasadd_readvariableop_resource:LP
:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource:LJ<
.model_conv1d_4_biasadd_readvariableop_resource:J@
,model_dense_1_matmul_readvariableop_resource:
??<
-model_dense_1_biasadd_readvariableop_resource:	??
,model_dense_2_matmul_readvariableop_resource:	?;
-model_dense_2_biasadd_readvariableop_resource:
identity??%model/Conv1D_1/BiasAdd/ReadVariableOp?1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_2/BiasAdd/ReadVariableOp?1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_3/BiasAdd/ReadVariableOp?1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_4/BiasAdd/ReadVariableOp?1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?$model/Dense_1/BiasAdd/ReadVariableOp?#model/Dense_1/MatMul/ReadVariableOp?$model/Dense_2/BiasAdd/ReadVariableOp?#model/Dense_2/MatMul/ReadVariableOp?
$model/Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_1/conv1d/ExpandDims/dim?
 model/Conv1D_1/conv1d/ExpandDims
ExpandDimsinput_1-model/Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2"
 model/Conv1D_1/conv1d/ExpandDims?
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[*
dtype023
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_1/conv1d/ExpandDims_1/dim?
"model/Conv1D_1/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[2$
"model/Conv1D_1/conv1d/ExpandDims_1?
model/Conv1D_1/conv1dConv2D)model/Conv1D_1/conv1d/ExpandDims:output:0+model/Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????[*
paddingVALID*
strides
2
model/Conv1D_1/conv1d?
model/Conv1D_1/conv1d/SqueezeSqueezemodel/Conv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????[*
squeeze_dims

?????????2
model/Conv1D_1/conv1d/Squeeze?
%model/Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:[*
dtype02'
%model/Conv1D_1/BiasAdd/ReadVariableOp?
model/Conv1D_1/BiasAddBiasAdd&model/Conv1D_1/conv1d/Squeeze:output:0-model/Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????[2
model/Conv1D_1/BiasAdd?
model/Conv1D_1/ReluRelumodel/Conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????[2
model/Conv1D_1/Relu?
#model/MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_1/ExpandDims/dim?
model/MaxPooling1D_1/ExpandDims
ExpandDims!model/Conv1D_1/Relu:activations:0,model/MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????[2!
model/MaxPooling1D_1/ExpandDims?
model/MaxPooling1D_1/MaxPoolMaxPool(model/MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:??????????[*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_1/MaxPool?
model/MaxPooling1D_1/SqueezeSqueeze%model/MaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:??????????[*
squeeze_dims
2
model/MaxPooling1D_1/Squeeze?
$model/Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_2/conv1d/ExpandDims/dim?
 model/Conv1D_2/conv1d/ExpandDims
ExpandDims%model/MaxPooling1D_1/Squeeze:output:0-model/Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????[2"
 model/Conv1D_2/conv1d/ExpandDims?
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[J*
dtype023
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_2/conv1d/ExpandDims_1/dim?
"model/Conv1D_2/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[J2$
"model/Conv1D_2/conv1d/ExpandDims_1?
model/Conv1D_2/conv1dConv2D)model/Conv1D_2/conv1d/ExpandDims:output:0+model/Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????J*
paddingVALID*
strides
2
model/Conv1D_2/conv1d?
model/Conv1D_2/conv1d/SqueezeSqueezemodel/Conv1D_2/conv1d:output:0*
T0*,
_output_shapes
:??????????J*
squeeze_dims

?????????2
model/Conv1D_2/conv1d/Squeeze?
%model/Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02'
%model/Conv1D_2/BiasAdd/ReadVariableOp?
model/Conv1D_2/BiasAddBiasAdd&model/Conv1D_2/conv1d/Squeeze:output:0-model/Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????J2
model/Conv1D_2/BiasAdd?
model/Conv1D_2/ReluRelumodel/Conv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????J2
model/Conv1D_2/Relu?
#model/MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_2/ExpandDims/dim?
model/MaxPooling1D_2/ExpandDims
ExpandDims!model/Conv1D_2/Relu:activations:0,model/MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????J2!
model/MaxPooling1D_2/ExpandDims?
model/MaxPooling1D_2/MaxPoolMaxPool(model/MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????IJ*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_2/MaxPool?
model/MaxPooling1D_2/SqueezeSqueeze%model/MaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????IJ*
squeeze_dims
2
model/MaxPooling1D_2/Squeeze?
$model/Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_3/conv1d/ExpandDims/dim?
 model/Conv1D_3/conv1d/ExpandDims
ExpandDims%model/MaxPooling1D_2/Squeeze:output:0-model/Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????IJ2"
 model/Conv1D_3/conv1d/ExpandDims?
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:JL*
dtype023
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_3/conv1d/ExpandDims_1/dim?
"model/Conv1D_3/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:JL2$
"model/Conv1D_3/conv1d/ExpandDims_1?
model/Conv1D_3/conv1dConv2D)model/Conv1D_3/conv1d/ExpandDims:output:0+model/Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????FL*
paddingVALID*
strides
2
model/Conv1D_3/conv1d?
model/Conv1D_3/conv1d/SqueezeSqueezemodel/Conv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????FL*
squeeze_dims

?????????2
model/Conv1D_3/conv1d/Squeeze?
%model/Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:L*
dtype02'
%model/Conv1D_3/BiasAdd/ReadVariableOp?
model/Conv1D_3/BiasAddBiasAdd&model/Conv1D_3/conv1d/Squeeze:output:0-model/Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????FL2
model/Conv1D_3/BiasAdd?
model/Conv1D_3/ReluRelumodel/Conv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????FL2
model/Conv1D_3/Relu?
#model/MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_3/ExpandDims/dim?
model/MaxPooling1D_3/ExpandDims
ExpandDims!model/Conv1D_3/Relu:activations:0,model/MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????FL2!
model/MaxPooling1D_3/ExpandDims?
model/MaxPooling1D_3/MaxPoolMaxPool(model/MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????L*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_3/MaxPool?
model/MaxPooling1D_3/SqueezeSqueeze%model/MaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????L*
squeeze_dims
2
model/MaxPooling1D_3/Squeeze?
$model/Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_4/conv1d/ExpandDims/dim?
 model/Conv1D_4/conv1d/ExpandDims
ExpandDims%model/MaxPooling1D_3/Squeeze:output:0-model/Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????L2"
 model/Conv1D_4/conv1d/ExpandDims?
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:LJ*
dtype023
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_4/conv1d/ExpandDims_1/dim?
"model/Conv1D_4/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:LJ2$
"model/Conv1D_4/conv1d/ExpandDims_1?
model/Conv1D_4/conv1dConv2D)model/Conv1D_4/conv1d/ExpandDims:output:0+model/Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????J*
paddingVALID*
strides
2
model/Conv1D_4/conv1d?
model/Conv1D_4/conv1d/SqueezeSqueezemodel/Conv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????J*
squeeze_dims

?????????2
model/Conv1D_4/conv1d/Squeeze?
%model/Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype02'
%model/Conv1D_4/BiasAdd/ReadVariableOp?
model/Conv1D_4/BiasAddBiasAdd&model/Conv1D_4/conv1d/Squeeze:output:0-model/Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????J2
model/Conv1D_4/BiasAdd?
model/Conv1D_4/ReluRelumodel/Conv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????J2
model/Conv1D_4/Relu{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????V  2
model/flatten/Const?
model/flatten/ReshapeReshape!model/Conv1D_4/Relu:activations:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten/Reshape?
#model/Dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#model/Dense_1/MatMul/ReadVariableOp?
model/Dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Dense_1/MatMul?
$model/Dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/Dense_1/BiasAdd/ReadVariableOp?
model/Dense_1/BiasAddBiasAddmodel/Dense_1/MatMul:product:0,model/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Dense_1/BiasAdd?
model/Dense_1/ReluRelumodel/Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/Dense_1/Relu?
#model/Dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#model/Dense_2/MatMul/ReadVariableOp?
model/Dense_2/MatMulMatMul model/Dense_1/Relu:activations:0+model/Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/Dense_2/MatMul?
$model/Dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/Dense_2/BiasAdd/ReadVariableOp?
model/Dense_2/BiasAddBiasAddmodel/Dense_2/MatMul:product:0,model/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/Dense_2/BiasAdd?
IdentityIdentitymodel/Dense_2/BiasAdd:output:0&^model/Conv1D_1/BiasAdd/ReadVariableOp2^model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_2/BiasAdd/ReadVariableOp2^model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_3/BiasAdd/ReadVariableOp2^model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_4/BiasAdd/ReadVariableOp2^model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp%^model/Dense_1/BiasAdd/ReadVariableOp$^model/Dense_1/MatMul/ReadVariableOp%^model/Dense_2/BiasAdd/ReadVariableOp$^model/Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2N
%model/Conv1D_1/BiasAdd/ReadVariableOp%model/Conv1D_1/BiasAdd/ReadVariableOp2f
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_2/BiasAdd/ReadVariableOp%model/Conv1D_2/BiasAdd/ReadVariableOp2f
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_3/BiasAdd/ReadVariableOp%model/Conv1D_3/BiasAdd/ReadVariableOp2f
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_4/BiasAdd/ReadVariableOp%model/Conv1D_4/BiasAdd/ReadVariableOp2f
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2L
$model/Dense_1/BiasAdd/ReadVariableOp$model/Dense_1/BiasAdd/ReadVariableOp2J
#model/Dense_1/MatMul/ReadVariableOp#model/Dense_1/MatMul/ReadVariableOp2L
$model/Dense_2/BiasAdd/ReadVariableOp$model/Dense_2/BiasAdd/ReadVariableOp2J
#model/Dense_2/MatMul/ReadVariableOp#model/Dense_2/MatMul/ReadVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
#__inference_signature_wrapper_12678
input_1
unknown:[
	unknown_0:[
	unknown_1:[J
	unknown_2:J
	unknown_3:JL
	unknown_4:L
	unknown_5:LJ
	unknown_6:J
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_121622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
??
?
!__inference__traced_restore_13347
file_prefix6
 assignvariableop_conv1d_1_kernel:[.
 assignvariableop_1_conv1d_1_bias:[8
"assignvariableop_2_conv1d_2_kernel:[J.
 assignvariableop_3_conv1d_2_bias:J8
"assignvariableop_4_conv1d_3_kernel:JL.
 assignvariableop_5_conv1d_3_bias:L8
"assignvariableop_6_conv1d_4_kernel:LJ.
 assignvariableop_7_conv1d_4_bias:J5
!assignvariableop_8_dense_1_kernel:
??.
assignvariableop_9_dense_1_bias:	?5
"assignvariableop_10_dense_2_kernel:	?.
 assignvariableop_11_dense_2_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: @
*assignvariableop_21_adam_conv1d_1_kernel_m:[6
(assignvariableop_22_adam_conv1d_1_bias_m:[@
*assignvariableop_23_adam_conv1d_2_kernel_m:[J6
(assignvariableop_24_adam_conv1d_2_bias_m:J@
*assignvariableop_25_adam_conv1d_3_kernel_m:JL6
(assignvariableop_26_adam_conv1d_3_bias_m:L@
*assignvariableop_27_adam_conv1d_4_kernel_m:LJ6
(assignvariableop_28_adam_conv1d_4_bias_m:J=
)assignvariableop_29_adam_dense_1_kernel_m:
??6
'assignvariableop_30_adam_dense_1_bias_m:	?<
)assignvariableop_31_adam_dense_2_kernel_m:	?5
'assignvariableop_32_adam_dense_2_bias_m:@
*assignvariableop_33_adam_conv1d_1_kernel_v:[6
(assignvariableop_34_adam_conv1d_1_bias_v:[@
*assignvariableop_35_adam_conv1d_2_kernel_v:[J6
(assignvariableop_36_adam_conv1d_2_bias_v:J@
*assignvariableop_37_adam_conv1d_3_kernel_v:JL6
(assignvariableop_38_adam_conv1d_3_bias_v:L@
*assignvariableop_39_adam_conv1d_4_kernel_v:LJ6
(assignvariableop_40_adam_conv1d_4_bias_v:J=
)assignvariableop_41_adam_dense_1_kernel_v:
??6
'assignvariableop_42_adam_dense_1_bias_v:	?<
)assignvariableop_43_adam_dense_2_kernel_v:	?5
'assignvariableop_44_adam_dense_2_bias_v:
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv1d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv1d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv1d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv1d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv1d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv1d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv1d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv1d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv1d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv1d_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv1d_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv1d_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv1d_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
(__inference_Conv1D_3_layer_call_fn_12953

inputs
unknown:JL
	unknown_0:L
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????FL*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_122762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????FL2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????IJ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????IJ
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_12171

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_12736

inputs
unknown:[
	unknown_0:[
	unknown_1:[J
	unknown_2:J
	unknown_3:JL
	unknown_4:L
	unknown_5:LJ
	unknown_6:J
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_125092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????;
Dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?i
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?e
_tf_keras_network?e{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 91, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 74, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 76, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_3", "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 74, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_4", "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 136, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["Dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}, "shared_object_id": 23, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 91, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 74, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 76, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_3", "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 74, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_4", "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 136, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 22}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}, "shared_object_id": 25}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 26}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010600000387057662, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "Conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 91, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 1]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "MaxPooling1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 28}}
?


kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 74, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 91}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 149, 91]}}
?
"	variables
#regularization_losses
$trainable_variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "MaxPooling1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 30}}
?


&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv1D_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 76, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 74}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 73, 74]}}
?
,	variables
-regularization_losses
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "MaxPooling1D_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 32}}
?


0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv1D_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 74, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 76}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17, 76]}}
?
6	variables
7regularization_losses
8trainable_variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 34}}
?	

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "Dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 136, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1110}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1110]}}
?	

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "Dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 136}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 136]}}
?
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_ratem?m?m?m?&m?'m?0m?1m?:m?;m?@m?Am?v?v?v?v?&v?'v?0v?1v?:v?;v?@v?Av?"
	optimizer
v
0
1
2
3
&4
'5
06
17
:8
;9
@10
A11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
&4
'5
06
17
:8
;9
@10
A11"
trackable_list_wrapper
?
	variables
Klayer_regularization_losses
regularization_losses
Lmetrics
trainable_variables
Mnon_trainable_variables
Nlayer_metrics

Olayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#[2Conv1D_1/kernel
:[2Conv1D_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Player_regularization_losses
	variables
regularization_losses
Qmetrics
trainable_variables
Rnon_trainable_variables
Slayer_metrics

Tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ulayer_regularization_losses
	variables
regularization_losses
Vmetrics
trainable_variables
Wnon_trainable_variables
Xlayer_metrics

Ylayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#[J2Conv1D_2/kernel
:J2Conv1D_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Zlayer_regularization_losses
	variables
regularization_losses
[metrics
 trainable_variables
\non_trainable_variables
]layer_metrics

^layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
_layer_regularization_losses
"	variables
#regularization_losses
`metrics
$trainable_variables
anon_trainable_variables
blayer_metrics

clayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#JL2Conv1D_3/kernel
:L2Conv1D_3/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
dlayer_regularization_losses
(	variables
)regularization_losses
emetrics
*trainable_variables
fnon_trainable_variables
glayer_metrics

hlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ilayer_regularization_losses
,	variables
-regularization_losses
jmetrics
.trainable_variables
knon_trainable_variables
llayer_metrics

mlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#LJ2Conv1D_4/kernel
:J2Conv1D_4/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
nlayer_regularization_losses
2	variables
3regularization_losses
ometrics
4trainable_variables
pnon_trainable_variables
qlayer_metrics

rlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
slayer_regularization_losses
6	variables
7regularization_losses
tmetrics
8trainable_variables
unon_trainable_variables
vlayer_metrics

wlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2Dense_1/kernel
:?2Dense_1/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
xlayer_regularization_losses
<	variables
=regularization_losses
ymetrics
>trainable_variables
znon_trainable_variables
{layer_metrics

|layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2Dense_2/kernel
:2Dense_2/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
}layer_regularization_losses
B	variables
Cregularization_losses
~metrics
Dtrainable_variables
non_trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 37}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 26}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:([2Adam/Conv1D_1/kernel/m
 :[2Adam/Conv1D_1/bias/m
*:([J2Adam/Conv1D_2/kernel/m
 :J2Adam/Conv1D_2/bias/m
*:(JL2Adam/Conv1D_3/kernel/m
 :L2Adam/Conv1D_3/bias/m
*:(LJ2Adam/Conv1D_4/kernel/m
 :J2Adam/Conv1D_4/bias/m
':%
??2Adam/Dense_1/kernel/m
 :?2Adam/Dense_1/bias/m
&:$	?2Adam/Dense_2/kernel/m
:2Adam/Dense_2/bias/m
*:([2Adam/Conv1D_1/kernel/v
 :[2Adam/Conv1D_1/bias/v
*:([J2Adam/Conv1D_2/kernel/v
 :J2Adam/Conv1D_2/bias/v
*:(JL2Adam/Conv1D_3/kernel/v
 :L2Adam/Conv1D_3/bias/v
*:(LJ2Adam/Conv1D_4/kernel/v
 :J2Adam/Conv1D_4/bias/v
':%
??2Adam/Dense_1/kernel/v
 :?2Adam/Dense_1/bias/v
&:$	?2Adam/Dense_2/kernel/v
:2Adam/Dense_2/bias/v
?2?
 __inference__wrapped_model_12162?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_1??????????
?2?
%__inference_model_layer_call_fn_12374
%__inference_model_layer_call_fn_12707
%__inference_model_layer_call_fn_12736
%__inference_model_layer_call_fn_12565?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_12815
@__inference_model_layer_call_and_return_conditional_losses_12894
@__inference_model_layer_call_and_return_conditional_losses_12603
@__inference_model_layer_call_and_return_conditional_losses_12641?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_Conv1D_1_layer_call_fn_12903?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_12919?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_MaxPooling1D_1_layer_call_fn_12177?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_12171?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
(__inference_Conv1D_2_layer_call_fn_12928?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_12944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_MaxPooling1D_2_layer_call_fn_12192?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_12186?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
(__inference_Conv1D_3_layer_call_fn_12953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_12969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_MaxPooling1D_3_layer_call_fn_12207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_12201?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
(__inference_Conv1D_4_layer_call_fn_12978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_12994?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_12999?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_13005?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Dense_1_layer_call_fn_13014?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense_1_layer_call_and_return_conditional_losses_13025?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Dense_2_layer_call_fn_13034?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense_2_layer_call_and_return_conditional_losses_13044?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_12678input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_12919f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????[
? ?
(__inference_Conv1D_1_layer_call_fn_12903Y4?1
*?'
%?"
inputs??????????
? "???????????[?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_12944f4?1
*?'
%?"
inputs??????????[
? "*?'
 ?
0??????????J
? ?
(__inference_Conv1D_2_layer_call_fn_12928Y4?1
*?'
%?"
inputs??????????[
? "???????????J?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_12969d&'3?0
)?&
$?!
inputs?????????IJ
? ")?&
?
0?????????FL
? ?
(__inference_Conv1D_3_layer_call_fn_12953W&'3?0
)?&
$?!
inputs?????????IJ
? "??????????FL?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_12994d013?0
)?&
$?!
inputs?????????L
? ")?&
?
0?????????J
? ?
(__inference_Conv1D_4_layer_call_fn_12978W013?0
)?&
$?!
inputs?????????L
? "??????????J?
B__inference_Dense_1_layer_call_and_return_conditional_losses_13025^:;0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_Dense_1_layer_call_fn_13014Q:;0?-
&?#
!?
inputs??????????
? "????????????
B__inference_Dense_2_layer_call_and_return_conditional_losses_13044]@A0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_Dense_2_layer_call_fn_13034P@A0?-
&?#
!?
inputs??????????
? "???????????
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_12171?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_1_layer_call_fn_12177wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_12186?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_2_layer_call_fn_12192wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_12201?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_3_layer_call_fn_12207wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
 __inference__wrapped_model_12162x&'01:;@A5?2
+?(
&?#
input_1??????????
? "1?.
,
Dense_2!?
Dense_2??????????
B__inference_flatten_layer_call_and_return_conditional_losses_13005]3?0
)?&
$?!
inputs?????????J
? "&?#
?
0??????????
? {
'__inference_flatten_layer_call_fn_12999P3?0
)?&
$?!
inputs?????????J
? "????????????
@__inference_model_layer_call_and_return_conditional_losses_12603t&'01:;@A=?:
3?0
&?#
input_1??????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_12641t&'01:;@A=?:
3?0
&?#
input_1??????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_12815s&'01:;@A<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_12894s&'01:;@A<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_12374g&'01:;@A=?:
3?0
&?#
input_1??????????
p 

 
? "???????????
%__inference_model_layer_call_fn_12565g&'01:;@A=?:
3?0
&?#
input_1??????????
p

 
? "???????????
%__inference_model_layer_call_fn_12707f&'01:;@A<?9
2?/
%?"
inputs??????????
p 

 
? "???????????
%__inference_model_layer_call_fn_12736f&'01:;@A<?9
2?/
%?"
inputs??????????
p

 
? "???????????
#__inference_signature_wrapper_12678?&'01:;@A@?=
? 
6?3
1
input_1&?#
input_1??????????"1?.
,
Dense_2!?
Dense_2?????????