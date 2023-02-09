using Plots, CSV, DataFrames
plotly()
Plots.default(show=true)
histWorldCup = CSV.read("data/world-cup/WorldCupsSummary.csv", DataFrame)

#=
1. 历年现场观众人数变化趋势
2. 参赛队伍数变化趋势
3. 历年进球数变化趋势
4. 历史上夺冠次数最多的国家队是哪支？
5. 夺冠队伍所在洲分析
6. 哪些国家队能经常打入决赛/半决赛？
7. 进入决赛的队伍夺冠概率是多少？
8. 东道主（主办国）进入决赛/半决赛大吗？
=#

# 历年现场观众人数变化趋势

# 数据预处理
let columns = names(histWorldCup)
  for column in columns
    replace!(histWorldCup[!, column], "Germany FR" => "Germany")
  end
end

let title = "Attendance Number"
    x = histWorldCup.Attendance
    y = histWorldCup.Year
    xticks = [500000, 1000000, 1500000, 2000000, 250000000, 3000000, 35000000, 4000000]
    yticks = histWorldCup.Year

    scatter(x, y, xticks=(:all, xticks), yticks=yticks, title=title) |> display
end

# 参赛队伍数变化趋势
let title = "QUalifiedTeams Number"
    x = histWorldCup.QualifiedTeams
    y = histWorldCup.Year
    xticks = [0, 16, 24, 32, 48]
    yticks = histWorldCup.Year

    scatter(x, y, xticks=(:all, xticks), yticks=yticks, title=title) |> display
end

# 历年进球数变化趋势
let title = "Goals Number"
    x = histWorldCup.GoalsScored
    y = histWorldCup.Year
    xticks = [50, 75, 100, 125, 150, 175, 200]
    yticks = histWorldCup.Year

    scatter(x, y, xticks=(:all, xticks), yticks=yticks, title=title) |> display
end

# 夺冠次数分析
import StatsBase: countmap
let title = "Champion Number Statistic"
    x = histWorldCup.Winner
    cmap = countmap(x)
    xs = convert(Vector{String}, collect(keys(cmap)))
    ys = collect(values(cmap))

    bar(xs, ys, title=title) |> display
end

# 半决赛（4强）队伍次数统计
cmap1 = countmap(histWorldCup.Winner)
cmap2 = countmap(histWorldCup.Second)
cmap3 = countmap(histWorldCup.Third)
cmap4 = countmap(histWorldCup.Fourth)

countries = DataFrame()
_countries = cat(collect(keys(cmap1)),
                collect(keys(cmap2)),
                collect(keys(cmap3)),
                collect(keys(cmap4)), dims=1) |> unique |> sort

countries[!, :Index] = _countries
fn(country::AbstractString, cmap::Dict{<:AbstractString,Int}) = begin
    if !haskey(cmap, country)
        return 0
    else
        return cmap[country]
    end
end

countries[!, :Winner] = map(x -> fn(x, cmap1), countries[!, :Index])
countries[!, :Second] = map(x -> fn(x, cmap2), countries[!, :Index])
countries[!, :Third] = map(x -> fn(x, cmap3), countries[!, :Index])
countries[!, :Fourth] = map(x -> fn(x, cmap4), countries[!, :Index])

countries[!, :SemiFinal] = countries[!, :Winner] .+ countries[!, :Second] .+ countries[!, :Third] .+ countries[!, :Fourth]
countries[!, :Final] = countries[!, :Winner] .+ countries[!, :Second]

let
    x = countries[!, :Index]
    y = countries[!, :SemiFinal]
    title = "SemiFinal Statistic"
    bar(x, y, title=title, xrotation=45, xticks=:all) |> display
end

# 决赛队伍次数统计
filterfn = [:Winner, :Second] => (winner, second) -> !((winner == 0) & (second == 0) != 0)
finalist = filter(filterfn, countries)

let x = finalist[!, :Index]
  y = finalist[!, :Final]
  title = "Final Statistic"
  bar(x, y, title = title, xticks = :all, xrotation = 45) |> display
end

# 进入决赛后夺冠以来分析
let
  finalist[!, :ChampionProb] = finalist[!, :Winner] ./ finalist[:, :Final]
  indexs = (finalist[!, :Second] .> 0) .| (finalist[!, :Winner] .> 0)
  ratios = finalist[indexs, :]

  x = ratios[!, :Index]
  y = ratios[!, :ChampionProb]
  title = "Percentage of winning reaching the final"
  bar(x, y, title = title, xticks = :all, xrotation = 45) |> display
end

# 夺冠队伍所在大洲分布
let cmap = countmap(histWorldCup.WinnerContinent)
  xs = collect(keys(cmap))
  ys = collect(values(cmap))
  title = "Champion Continent Numbers"
  bar(xs, ys, title = title, xticks = :all) |> display

  summary = reduce(+, ys)
  _values = map(x -> x / summary, ys)
  title = "Champion Continent Ratios"
  pie(xs, _values, title = title) |> display
end

# 东道主进入半决赛/决赛/夺冠概率统计
let hostTop4 = map(row -> in(row.HostCountry, [row.Winner, row.Second, row.Third, row.Fourth]) ? 1 : 0, eachrow(histWorldCup))
  cmap = countmap(hostTop4)
  x = collect(keys(cmap))
  y = collect(values(cmap))
  title = "Host in Top4"
  bar(x, y, title = title, xticks = :all) |> display

  summary = reduce(+, y)
  _values = map(x -> x / summary, y)

  title = "Percentage"
  pie(x, _values, title = title) |> display
end

# 东道主进入决赛概率
let hostTop2 = map(row -> in(row.HostCountry, [row.Winner, row.Second]) ? 1 : 0, eachrow(histWorldCup))
  cmap = countmap(hostTop2)
  x = collect(keys(cmap))
  y = collect(values(cmap))
  title = "Host in Top2"
  bar(x, y, title = title, xticks = :all) |> display

  summary = reduce(+, y)
  _values = map(x -> x / summary, y)

  title = "Percentage"
  pie(x, _values, title = title) |> display
end
# 东道主夺冠概率
let hostWinner = map(row -> row.HostCountry == row.Winner ? 1 : 0, eachrow(histWorldCup))
  cmap = countmap(hostWinner)
  x = collect(keys(cmap))
  y = collect(values(cmap))
  title = "Host in Winner"
  bar(x, y, title = title, xticks = :all) |> display

  summary = reduce(+, y)
  _values = map(x -> x / summary, y)

  title = "Percentage"
  pie(x, _values, title = title) |> display
end

# 分析世界杯比赛信息表
# PROBLEM 在 Attendance 字段有 missing 数据，然而在教程中我没有找到相关 fillmissing 的语句
matches = CSV.read("data/world-cup/WorldCupMatches.csv", DataFrame)

# 中国队参加的比赛
let indexs = (matches[!, Symbol("Away Team Name")] .== "China PR") .| (matches[!, Symbol("Home Team Name")] .== "China PR")
  matches[indexs, :]

end

# 统一联邦德国和德国
let columns = names(matches)
  for column in columns
    replace!(matches[!, column], "Germany FR" => "Germany")
  end
end

# 类型转化
matches[!, "Home Team Goals"] = convert(Vector{Int}, matches[!, "Home Team Goals"])
matches[!, "Away Team Goals"] = convert(Vector{Int}, matches[!, "Away Team Goals"])
matches[!, "Result"] = map((x, y) -> "$(x) - $(y)", matches[!, "Home Team Goals"], matches[!, "Away Team Goals"])

# 现场观赛人数分析
# PROBLEM there is missing data in `attendance` column
let top5Attendance = first(sort(matches, [order(:Attendance, rev=true)]), 5)
  top5Attendance[!, :VS] = map((x, y) -> "$(x) VS $(y)", top5Attendance[!, "Home Team Name"], top5Attendance[!, "Away Team Name"])
  x = top5Attendance[!, :VS]
  y = top5Attendance[!, :Attendance]
  bar(x, y) |> display
end

# 比赛进球数分析
matches[!, :TotalGoals] = matches[!, "Home Team Goals"] .+ matches[!, "Away Team Goals"]
matches[!, :VS] = map((x, y) -> "$(x) VS $(y)", matches[!, "Home Team Name"], matches[!, "Away Team Name"])

let top10Goals = first(sort(matches, [order(:TotalGoals, rev=true)]), 10)
  top10Goals[!, :VS] = map((x, y) -> "$(x) VS $(y)", top10Goals[!, "Home Team Name"], top10Goals[!, "Away Team Name"])
  top10Goals[!, :TotalGoalsStr] = map(x -> "$(x) goals scored", top10Goals[!, "TotalGoals"])
  top10Goals[!, "Home Team Goals"] = convert(Vector{Int}, top10Goals[!, "Home Team Goals"])
  top10Goals[!, "Away Team Goals"] = convert(Vector{Int}, top10Goals[!, "Away Team Goals"])
  top10Goals[!, "Result"] = map((x, y) -> "$(x) - $(y)", top10Goals[!, "Home Team Goals"], top10Goals[!, "Away Team Goals"])

  x = top10Goals[!, "VS"]
  y = top10Goals[!, "TotalGoals"]
  title = "Top10 Goals Match"

  bar(x, y, title = title, xrotation = 90, xticks = :all) |> display
end

# 我们再来分析比赛分差最大的比赛
let
  matches[!, "DifferenceGoals"] = abs.(matches[!, "Home Team Goals"] .- matches[!, "Away Team Goals"])
  top10Difference = first(sort(matches, [order("DifferenceGoals", rev = true)]), 10)
  top10Difference[!, "DifferenceGoals"] = convert(Vector{Int}, top10Difference[!, "DifferenceGoals"])
  top10Difference[!, "DifferenceGoalsStr"] = map(x -> "$(x) goals difference", top10Difference[!, "DifferenceGoals"])
  top10Difference[!, "Result"] = map((x, y) -> "$(x) - $(y)", top10Difference[!, "Home Team Goals"], top10Difference[!, "Away Team Goals"])

  x = top10Difference[!, "VS"]
  y = top10Difference[!, "DifferenceGoals"]
  title = "Top10 Biggest Difference Matches"


  bar(x, y, title = title, xrotation = 90) |> display
end

# 进球数分析
using StatsPlots
let columns = names(matches)
  for column in columns
    replace!(matches[!, column], "Germany FR" => "Germany")
  end

  listCountries = unique(matches[!, "Home Team Name"])
  listHome = Int[]
  listAway = Int[]
  for country in listCountries
    indexs = matches[!, "Home Team Name"] .== country
    goalsHome = reduce(+, matches[!, "Home Team Goals"][indexs])
    push!(listHome, goalsHome)

    indexs = matches[!, "Away Team Name"] .== country
    goalsAway = reduce(+, matches[!, "Away Team Goals"][indexs])
    push!(listAway, goalsAway)
  end

  df = DataFrame(Country = listCountries, TotalHomeGoals = listHome, TotalAwayGoals = listAway)
  df[!, "TotalGoals"] = df[!, "TotalHomeGoals"] .+ df[!, "TotalAwayGoals"]
  mostGoals = first(sort(df, [order(:TotalGoals, rev=true)]), 10)

  x = mostGoals[!, "Country"]
  y = select(mostGoals, ["TotalHomeGoals", "TotalAwayGoals", "TotalGoals"]) |> Matrix 
  groupedbar(x, y, xrotation = 90, labels = ["TotalHomeGoals" "TotalAwayGoals" "TotalGoals"]) |> display
end

# 失球数分析
let finalista = finalist[!, :Index]
  goalsConcededHome = Int[]
  goalsConcededAway = Int[]
  match1 = Int[]
  match2 = Int[]

  for country in finalista
    indexs = matches[!, "Home Team Name"] .== country
    goalsConcHome = reduce(+, matches[!, "Away Team Goals"][indexs])
    push!(goalsConcededHome, goalsConcHome)
    counted1 = reduce(+, indexs)
    
    indexs = matches[!, "Away Team Name"] .== country
    goalsConcAway = reduce(+, matches[!, "Home Team Goals"][indexs])
    push!(goalsConcededAway, goalsConcAway)
    counted2 = reduce(+, indexs)

    push!(match1, counted1)
    push!(match2, counted2)
  end

  df = DataFrame(Country = finalista, GoalsConcededHome = goalsConcededHome, GoalsConcededAway = goalsConcededAway, MatchesHome = match1, MatchesAway = match2)
  df[!, "TotalMatches"] = df[!, "MatchesHome"] .+ df[!, "MatchesAway"]
  df[!, "TotalGoalsConceded"] = df[!, "GoalsConcededHome"] .+ df[!, "GoalsConcededAway"]
  df[!, "GoalMatchRate"] = round.(df[!, "TotalGoalsConceded"] ./ df[!, "TotalMatches"], digits = 2)

  goalsConceded = first(sort(df, [order("GoalMatchRate", rev=true)]), 10)

  x = goalsConceded[!, "Country"]
  y = goalsConceded[!, "TotalGoalsConceded"]
  bp1 = bar(x, y, xrotation = 45)

  x = goalsConceded[!, "Country"]
  y = goalsConceded[!, "GoalMatchRate"]
  bp2 = bar(x, y, xrotation = 45)

  plot(bp1, bp2) |> display
  
end

