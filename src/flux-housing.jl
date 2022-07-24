using Flux
using Flux: gradient
using Flux.Optimise: update!
using DelimitedFiles, Statistics
using Parameters: @with_kw

@with_kw mutable struct HyperParams
  lr::Float64 = 0.1
  splitRatio::Float64 = 0.1
end

function getProcessedData(args)
  # download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data", "data/housing.data")
  rawdata = readdlm("data/housing.data")
  splitRatio = args.splitRatio

  x = rawdata[1:13, :]
  y = rawdata[14:14, :]

  x = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

  splitIndex = floor(Int, size(x, 2) * splitRatio)
  x_train = x[:, 1:splitIndex]
  y_train = y[:, 1:splitIndex]
  x_test = x[:, splitIndex + 1 : size(x, 2)]
  y_test = y[:, splitIndex + 1 : size(x, 2)]

  train_data = (x_train, y_train)
  test_data = (x_test, y_test)

  return train_data, test_data
end

mutable struct Model
  W::AbstractArray
  b::AbstractVector
end

predict(x, m) = m.W * x .+ m.b

meanSquaredError(ŷ, y) = reduce(+, map(x -> x ^ 2, ŷ .- y)) / size(y, 2)

function train(; kwargs...)
  args = HyperParams(; kwargs...)

  (x_train, y_train), (x_test, y_test) = getProcessedData(args)

  model = Model(randn(1, 13), [0.])
  loss(x, y) = meanSquaredError(predict(x, model), y)

  learningRate = args.lr
  theta = Flux.params(model.W, model.b)

  for epoch in 1:500
    g = gradient(() -> loss(x_train, y_train), theta)
    for x in theta
      update!(x, g[x] * learningRate)
    end

    if epoch % 100 == 0
      @show loss(x_train, y_train)
    end
  end

  error = meanSquaredError(predict(x_test, model), y_test)
  println(error)
end

train()