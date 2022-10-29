using Flux, MLJFlux
import MLJFlux

struct VGGNet
    batchsize::Int
end

function MLJFlux.build(vggnet::VGGNet, rng, nin, nout, nchannels)
    return @autosize (nin..., nchannels, vggnet.batchsize) Chain(
        Conv((3, 3), nchannels => 64, pad = SamePad()),
        BatchNorm(64, relu),

        Conv((3, 3), _ => 64, pad = SamePad()),
        BatchNorm(_, relu),
        MaxPool((2, 2), stride = 2, pad = SamePad()),
        Dropout(0.2),

        Conv((3, 3), _ => 128, pad = SamePad()),
        BatchNorm(_, relu),

        Conv((3, 3), _ => 128, pad = SamePad()),
        BatchNorm(_, relu),
        MaxPool((2, 2), stride = 2, pad = SamePad()),
        Dropout(0.2),

        Conv((3, 3), _ => 256, pad = SamePad()),
        BatchNorm(_, relu),

        Conv((3, 3), _ => 256, pad = SamePad()),
        BatchNorm(_, relu),

        Conv((3, 3), _ => 256, pad = SamePad()),
        BatchNorm(_, relu),
        MaxPool((2, 2), stride = 2, pad = SamePad()),
        Dropout(0.2),

        Conv((3, 3), _ => 512, pad = SamePad()),
        BatchNorm(_, relu),

        Conv((3, 3), _ => 512, pad = SamePad()),
        BatchNorm(_, relu),

        Conv((3, 3), _ => 512, pad = SamePad()),
        BatchNorm(_, relu),
        MaxPool((2, 2), stride = 2, pad = SamePad()),
        Dropout(0.2),

        Conv((3, 3), _ => 512, pad = SamePad()),
        BatchNorm(_, relu),

        Conv((3, 3), _ => 512, pad = SamePad()),
        BatchNorm(_, relu),

        Conv((3, 3), _ => 512, pad = SamePad()),
        BatchNorm(_, relu),
        MaxPool((2, 2), stride = 2, pad = SamePad()),
        Dropout(0.2),

        flatten,
        
        Dense(_ => 512, relu),
        Dropout(0.2),
        Dense(_ => 512, relu),
        Dropout(0.2),
        Dense(_ => nout)
    )
end

function buildVGGNet(;batchsize::Int, epochs::Int, lambda::Float32, alpha::Float32, usegpu = false)
    return ImageClassifier(
        builder = VGGNet(batchsize),
        batch_size = batchsize,
        epochs = epochs,
        rng = StableRNG(1234),
        lambda = lambda,
        alpha = alpha,
        acceleration = usegpu ? CUDALibs() : CPU1()
    )
end