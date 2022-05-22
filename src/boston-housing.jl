using MLJ, CSV, StableRNGs, MLJLinearModels, Plots, StatsPlots
import DataFrames: DataFrame, select, describe
using Statistics


dataTrain = CSV.read("src/data/boston-housing/train.csv", DataFrame)
dataTest = CSV.read("src/data/boston-housing/test.csv", DataFrame)

plotly()

let column = :CentralAir
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    boxplot(columnX, columnY) |> display
end

let column = :OverallQual
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    boxplot(columnX, columnY) |> display
end

let column = :YearBuilt
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    boxplot(columnX, columnY, size=(2600, 1200)) |> display
end

let column = :YearBuilt
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    boxplot(columnX, columnY, size=(2600, 1200)) |> display
    scatter(columnX, columnY, ylim=(0, 800000), size=(1500, 1000)) |> display
end

let column = :Neighborhood
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    boxplot(columnX, columnY, size = (1300, 600)) |> display
end

let column = :LotArea
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    scatter(columnX, columnY) |> display
end

let column = :GrLivArea
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    scatter(columnX, columnY) |> display
end

let column = :TotalBsmtSF
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    scatter(columnX, columnY) |> display
end

let column = :MiscVal
    columnY = dataTrain[!, :SalePrice]
    columnX = dataTrain[!, column]
    scatter(columnX, columnY) |> display
end

let columns = [:GarageArea, :GarageCars]
    columnY = dataTrain[!, :SalePrice]
    columnXs = map(column -> dataTrain[!, column], columns)

    for columnX in columnXs
	scatter(columnX, columnY) |> display
    end
end

let _schema = schema(dataTrain)
    _names = _schema.names
    _scitypes = _schema.scitypes
    indexs = collect(map(x -> x == Count || x == Continuous, _scitypes))
    columns = _names[indexs] |> collect
    _data = select(dataTrain, columns)
    _corr = cor(Matrix(_data))
    labels = string.(columns)
    heatmap(labels, labels, _corr, xrotation = -90, size = figureSize, xticks = :all, yticks = :all) |> display
end

let _schema = schema(dataTrain)
    _names = _schema.names
    _scitypes = _schema.scitypes
    indexs = collect(map(x -> x == Count || x == Continuous, _scitypes))
    columns = _names[indexs] |> collect
    labels = string.(columns)
    _data = select(dataTrain, columns)
    _corr = cor(Matrix(_data))

    _dataframe = DataFrame(_corr, columns)
    nlarget = _dataframe[partialsortperm(_dataframe[!, :SalePrice], 1:10, rev=true), :]

    heatmap(Matrix(nlarget), xrotation = -90, size = figureSize, xticks = :all, yticks = :all, aspect_ratio = :equal)

    nrow, ncol = size(_corr)
    fontsize = 15

    fn(tuple) = (tuple[1], tuple[2], text(round(_corr[tuple[1], tuple[2]], digits = 2), fontsize, :white, :center))
    ann = map(fn, Iterators.product(1:nrow, 1:ncol) |> collect |> vec)

    annotate!(ann, linecolor = :white) |> display
end

columns = [:OverallQual, :GrLivArea, :GarageCars, :TotalBsmtSF, :FullBath, :TotRmsAbvGrd, :YearBuilt]

trainTransformModel = Pipeline(
    FeatureSelector(features = columns),
    dataframe -> coerce(dataframe, Count => Continuous))

processFeature!(dataframe::DataFrame) = begin
    dataframe[!, :GarageCars] = replace(dataframe[!, :GarageCars], "NA" => missing)
    dataframe[!, :GarageCars] = map(x -> ismissing(x) ? x : parse(Float64, x), dataframe[!, :GarageCars])
    dataframe[!, :TotalBsmtSF] = replace(dataframe[!, :TotalBsmtSF], "NA" => missing)
    dataframe[!, :TotalBsmtSF] = map(x -> ismissing(x) ? x : parse(Float64, x), dataframe[!, :TotalBsmtSF])

    coerce!(dataframe, Count => Continuous)
    return dataframe
end

testTransformModel = Pipeline(
    FeatureSelector(features = columns),
    processFeature!,
    FillImputer(features = columns),
    # Standardizer(features = columns)
)

trainTransformMach = machine(trainTransformModel, dataTrain)
testTransformMach = machine(testTransformModel, dataTest)
fit!(trainTransformMach)
fit!(testTransformMach)

rng = StableRNG(1234)
cv = CV(nfolds = 6, rng = rng)
tuning = Grid(resolution=10, rng = rng)


transformedDataTrain = transform(trainTransformMach, dataTrain)
transformedDataTest = transform(testTransformMach, dataTest)

X = transformedDataTrain
y = coerce(dataTrain[!, :SalePrice], Continuous)
train, test = partition(eachindex(y), 0.8, rng=rng)


# MODULE try Ridge
ridge = RidgeRegressor()
rangeLambda = range(ridge, :lambda, lower = 0.1, upper = 10.0, scale=:log)


tunedModel = TunedModel(model = ridge,
			range = [rangeLambda],
			measure = rms,
			resampling = cv,
			tuning = tuning)
tunedMach = machine(tunedModel, X, y)
fit!(tunedMach, rows = train)

evaluate!(tunedMach, resampling = cv, measure = [rms, l1], rows = test)

LGBMRegressor = @load LGBMRegressor
lgb = LGBMRegressor()
lgbm = machine(lgb, X, y)
boostRange = range(lgb, :num_iterations, lower = 2, upper = 500)
rangeLeaf = range(lgb, :min_data_in_leaf, lower = 1, upper = 50)
rangeIteration = range(lgb, :num_iterations, lower = 50, upper = 100)
rangeMinData = range(lgb, :min_data_in_leaf, lower = 2, upper = 10)
rangeLearningRate = range(lgb, :learning_rate, lower = 0.1, upper = 1)

tunedModel = TunedModel(model = lgb,
			tuning = Grid(resolution = 5, rng = rng),
			resampling = cv,
			ranges = [rangeIteration, rangeMinData, rangeLearningRate],
			measure = rms)

tunedMachine = machine(tunedModel, X, y)
fit!(tunedMachine, rows = train)
evaluate!(tunedMach, resampling = cv, measure = [rms, l1], rows = test)
predictions = predict(tunedMachine, transformedDataTest)
output = DataFrame(Id=dataTest.Id)
output[!, :SalePrice] = predictions
CSV.write("data/boston-housing/submission.csv", output)