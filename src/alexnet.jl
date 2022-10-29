using Flux, MLJFlux
import StableRNGs: StableRNG

struct AlexNet
    batchsize::Int
end

function MLJFlux.build(alexnet::AlexNet, rng, nin, nout, nchannels)
    return @autosize (nin..., nchannels, alexnet.batchsize) Chain(
        Conv((3, 3), nchannels => 96),
        BatchNorm(_, relu),
        MaxPool((3, 3), stride = 2),

        Conv((3, 3), _ => 256),
        BatchNorm(_, relu),
        MaxPool((3, 3), stride = 2),

        Conv((3, 3), _ => 384, relu, pad = SamePad()),
        Conv((3, 3), _ => 384, relu, pad = SamePad()),
        Conv((3, 3), _ => 256, relu, pad = SamePad()),
        MaxPool((3, 3), stride = 2),

        flatten,
        Dense(_ => 2048, relu),
        Dropout(0.5),
        Dense(_ => 2048, relu),
        Dropout(0.5),
        Dense(_ => nout)
    )
end


function buildAlexNet(;batchsize::Int, epochs::Int, lambda::Float32, alpha::Float32, usegpu = false)
    return ImageClassifier(
        builder = AlexNet(batchsize),
        batch_size = batchsize,
        epochs = epochs,
        rng = StableRNG(1234),
        lambda = lambda,
        alpha = alpha,
        acceleration = usegpu ? CUDALibs() : CPU1()
    )
end