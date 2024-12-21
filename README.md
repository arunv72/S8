{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red34\green45\blue53;}
{\*\expandedcolortbl;;\cssrgb\c17647\c23137\c27059;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}}{\leveltext\leveltemplateid1\'01\'00;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}}{\leveltext\leveltemplateid2\'01\'01;}{\levelnumbers\'01;}\fi-360\li1440\lin1440 }{\listname ;}\listid1}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa120\partightenfactor0

\f0\fs57\fsmilli28800 \cf2 \expnd0\expndtw0\kerning0
Session 8 Assignment Instructions\
\pard\pardeftab720\sa240\partightenfactor0

\fs32 \cf2 Write a new network that\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0\cf2 \kerning1\expnd0\expndtw0 {\listtext	1	}\expnd0\expndtw0\kerning0
Has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	2	}\expnd0\expndtw0\kerning0
total RF must be more than 44\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	3	}\expnd0\expndtw0\kerning0
one of the layers must use Depthwise Separable Convolution\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	4	}\expnd0\expndtw0\kerning0
one of the layers must use Dilated Convolution\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	5	}\expnd0\expndtw0\kerning0
use GAP (compulsory):- add FC after GAP to target #of classes (optional)\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	6	}\expnd0\expndtw0\kerning0
use augmentation library and apply:\
\pard\tx940\tx1440\pardeftab720\li1440\fi-1440\partightenfactor0
\ls1\ilvl1\cf2 \kerning1\expnd0\expndtw0 {\listtext	1	}\expnd0\expndtw0\kerning0
horizontal flip\
\ls1\ilvl1\kerning1\expnd0\expndtw0 {\listtext	2	}\expnd0\expndtw0\kerning0
shiftScaleRotate\
\ls1\ilvl1\kerning1\expnd0\expndtw0 {\listtext	3	}\expnd0\expndtw0\kerning0
coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0\cf2 \kerning1\expnd0\expndtw0 {\listtext	7	}\expnd0\expndtw0\kerning0
achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	8	}\expnd0\expndtw0\kerning0
make sure you're following code modularity (else 0 for full assignment)\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	9	}\expnd0\expndtw0\kerning0
upload to Github}