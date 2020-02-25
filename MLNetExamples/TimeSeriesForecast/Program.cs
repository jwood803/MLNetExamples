using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;

namespace TimeSeriesForecast
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<EnergyData>("./energy_hourly.csv",
                hasHeader: true, separatorChar: ',');

            var pipeline = context.Forecasting.ForecastBySsa(
                nameof(EnergyForecast.Forecast),
                nameof(EnergyData.Energy),
                windowSize: 5,
                seriesLength: 10,
                trainSize: 100,
                horizon: 4);

            var model = pipeline.Fit(data);

            var forecastingEngine = model.CreateTimeSeriesEngine<EnergyData, EnergyForecast>(context);

            var forecasts = forecastingEngine.Predict();

            foreach (var forecast in forecasts.Forecast)
            {
                Console.WriteLine(forecast);
            }

            Console.ReadLine();
        }
    }
}
