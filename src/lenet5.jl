using Flux, MLJFlux
import StableRNGs: StableRNG


struct LeNet5 
    batchsize::Int
end

function MLJFlux.build(lenet5::LeNet5, rng, nin, nout, nchannels)
    return @autosize (nin..., nchannels, lenet5.batchsize) Chain(
      Conv((5, 5), _ => 6, relu),
      MaxPool((2, 2)),
      Conv((5, 5), _ => 16, relu),
      MaxPool((2, 2)),
      Flux.flatten,
      Dense(_ => 120, relu),
      Dense(120 => 84, relu),
      Dense(84 => nout)
    )
end

function buildLeNet5(;batchsize::Int, epochs::Int, lambda::Float32, alpha::Float32, usegpu = false)
    return ImageClassifier(
        builder = LeNet5(batchsize),
        batch_size = batchsize,
        epochs = epochs,
        rng = StableRNG(1234),
        lambda = lambda,
        alpha = alpha,
        acceleration = usegpu ? CUDALibs() : CPU1()
    )
end