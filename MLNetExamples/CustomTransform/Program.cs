using Microsoft.ML;
using System;
using System.Collections.Generic;

namespace CustomTransform
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var sampleData = new List<InputData>
            {
                new InputData { Date = new DateTime(2019, 7, 19) },
                new InputData { Date = new DateTime(2019, 7, 6) },
                new InputData { Date = new DateTime(2019, 7, 2) },
                new InputData { Date = new DateTime(2019, 7, 14) },
            };

            var data = mlContext.Data.LoadFromEnumerable(sampleData);

            Action<InputData, MappingOutput> mapping = (input, output) =>
                output.IsWeekend = input.Date.DayOfWeek == DayOfWeek.Saturday || input.Date.DayOfWeek == DayOfWeek.Sunday;

            var pipeline = mlContext.Transforms.CustomMapping(mapping, "customMap");

            var transformedData = pipeline.Fit(data).Transform(data);

            var enumerableData = mlContext.Data.CreateEnumerable<NewData>(transformedData, reuseRowObject: true);

            foreach (var row in enumerableData)
            {
                Console.WriteLine($"{row.Date} - {row.IsWeekend}");
            }

            Console.ReadLine();
        }
    }
}
