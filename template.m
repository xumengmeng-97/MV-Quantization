%{ 
    -------- THIS IS AN AUTO-GENERATED NETWORK TEMPLATE --------
    NOTE: There is no guarantee for correctness. To start a 
    simulation, check the topology with the original network and 
    add necessary operations in the below code.

    (1). The shortcut connection is ignored in conversion.
    (2). Numeric type cast may be incorrect.
    ------------------------------------------------------------ 
%} 
% Author: Zhao Mingxin
% Date: 2021-03-10

function [im, stat] = template(nn, net, im)
    f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
    'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength', 32, ... 
     'ProductFractionLength', 0, 'SumWordLength', 32, 'SumFractionLength', 0); 
    t = numerictype('WordLength', 32, 'FractionLength', 0); 

% --- WARNING: Input is adjusted to [-128, 127].
% --- If your pre-processing is not like this,
% --- change it to what you used.
    im = fi(single(im)-128, t, f);

% --- Layer: root.features.0.conv.0
    im = nn.Conv2d(im, net{1}.Weight, t, f, [2, 2], 'VALID');
    im = nn.AddBias(im, net{1}.Bias, t, f);
    im = cast_int(im, net{1}.Mul, net{1}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.1.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{2}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{2}.Bias, t, f);
    im = cast_int(im, net{2}.Mul, net{2}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.1.conv.1.0
    im = nn.PointwiseConv2d(im, net{3}.Weight, t, f);
    im = nn.AddBias(im, net{3}.Bias, t, f);
    im = cast_int(im, net{3}.Mul, net{3}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.2.conv.0.0
    im = nn.DepthwiseConv2d(im, net{4}.Weight, t, f, [2, 2], 'VALID');
    im = nn.AddBias(im, net{4}.Bias, t, f);
    im = cast_int(im, net{4}.Mul, net{4}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.2.conv.1.0
    im = nn.PointwiseConv2d(im, net{5}.Weight, t, f);
    im = nn.AddBias(im, net{5}.Bias, t, f);
    im = cast_int(im, net{5}.Mul, net{5}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.3.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{6}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{6}.Bias, t, f);
    im = cast_int(im, net{6}.Mul, net{6}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.3.conv.1.0
    im = nn.PointwiseConv2d(im, net{7}.Weight, t, f);
    im = nn.AddBias(im, net{7}.Bias, t, f);
    im = cast_int(im, net{7}.Mul, net{7}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.4.conv.0.0
    im = nn.DepthwiseConv2d(im, net{8}.Weight, t, f, [2, 2], 'VALID');
    im = nn.AddBias(im, net{8}.Bias, t, f);
    im = cast_int(im, net{8}.Mul, net{8}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.4.conv.1.0
    im = nn.PointwiseConv2d(im, net{9}.Weight, t, f);
    im = nn.AddBias(im, net{9}.Bias, t, f);
    im = cast_int(im, net{9}.Mul, net{9}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.5.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{10}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{10}.Bias, t, f);
    im = cast_int(im, net{10}.Mul, net{10}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.5.conv.1.0
    im = nn.PointwiseConv2d(im, net{11}.Weight, t, f);
    im = nn.AddBias(im, net{11}.Bias, t, f);
    im = cast_int(im, net{11}.Mul, net{11}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.6.conv.0.0
    im = nn.DepthwiseConv2d(im, net{12}.Weight, t, f, [2, 2], 'VALID');
    im = nn.AddBias(im, net{12}.Bias, t, f);
    im = cast_int(im, net{12}.Mul, net{12}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.6.conv.1.0
    im = nn.PointwiseConv2d(im, net{13}.Weight, t, f);
    im = nn.AddBias(im, net{13}.Bias, t, f);
    im = cast_int(im, net{13}.Mul, net{13}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.7.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{14}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{14}.Bias, t, f);
    im = cast_int(im, net{14}.Mul, net{14}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.7.conv.1.0
    im = nn.PointwiseConv2d(im, net{15}.Weight, t, f);
    im = nn.AddBias(im, net{15}.Bias, t, f);
    im = cast_int(im, net{15}.Mul, net{15}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.8.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{16}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{16}.Bias, t, f);
    im = cast_int(im, net{16}.Mul, net{16}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.8.conv.1.0
    im = nn.PointwiseConv2d(im, net{17}.Weight, t, f);
    im = nn.AddBias(im, net{17}.Bias, t, f);
    im = cast_int(im, net{17}.Mul, net{17}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.9.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{18}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{18}.Bias, t, f);
    im = cast_int(im, net{18}.Mul, net{18}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.9.conv.1.0
    im = nn.PointwiseConv2d(im, net{19}.Weight, t, f);
    im = nn.AddBias(im, net{19}.Bias, t, f);
    im = cast_int(im, net{19}.Mul, net{19}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.10.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{20}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{20}.Bias, t, f);
    im = cast_int(im, net{20}.Mul, net{20}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.10.conv.1.0
    im = nn.PointwiseConv2d(im, net{21}.Weight, t, f);
    im = nn.AddBias(im, net{21}.Bias, t, f);
    im = cast_int(im, net{21}.Mul, net{21}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.11.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{22}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{22}.Bias, t, f);
    im = cast_int(im, net{22}.Mul, net{22}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.11.conv.1.0
    im = nn.PointwiseConv2d(im, net{23}.Weight, t, f);
    im = nn.AddBias(im, net{23}.Bias, t, f);
    im = cast_int(im, net{23}.Mul, net{23}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.12.conv.0.0
    im = nn.DepthwiseConv2d(im, net{24}.Weight, t, f, [2, 2], 'VALID');
    im = nn.AddBias(im, net{24}.Bias, t, f);
    im = cast_int(im, net{24}.Mul, net{24}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.12.conv.1.0
    im = nn.PointwiseConv2d(im, net{25}.Weight, t, f);
    im = nn.AddBias(im, net{25}.Bias, t, f);
    im = cast_int(im, net{25}.Mul, net{25}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.13.conv.0.0
    im = nn.ZeroPad2d(im, [1, 1]);
    im = nn.DepthwiseConv2d(im, net{26}.Weight, t, f, [1, 1], 'VALID');
    im = nn.AddBias(im, net{26}.Bias, t, f);
    im = cast_int(im, net{26}.Mul, net{26}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.features.13.conv.1.0
    im = nn.PointwiseConv2d(im, net{27}.Weight, t, f);
    im = nn.AddBias(im, net{27}.Bias, t, f);
    im = cast_int(im, net{27}.Mul, net{27}.Shift);
    im = nn.ReLU(im);

% --- Layer: root.0
    im = nn.PointwiseConv2d(im, net{28}.Weight, t, f);
    im = nn.AddBias(im, net{28}.Bias, t, f);
    im = cast_int(im, net{28}.Mul, net{28}.Shift);

    stat={};  % Collect desired intermediate results in stat.
end

function res = cast_int(im, mul, sft) 
%------ Uncomment to use intermediate results cast.------
%    im(im < -32768) = -32768;
%    im(im > 32767) = 32767;
%-------------------- Comment end. ----------------------
    im = im * mul;
    im = bitshift(im, -sft);
    im(im > 127) = 127;
    im(im < -128) = -128;
    res = im;
end
