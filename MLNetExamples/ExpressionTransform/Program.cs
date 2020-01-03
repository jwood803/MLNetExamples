using Microsoft.ML;
using System;
using System.Collections.Generic;

namespace ExpressionTransform
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var inputData = new List<SalaryInput>
            {
                new SalaryInput { IsManager = false, YearsExperience = 1f, Title = "Developer" },
                new SalaryInput { IsManager = true, YearsExperience = 9f, Title = "Director", NumberOfTeamsManaged = 2 },
                new SalaryInput { IsManager = false, YearsExperience = 4f, Title = "Analyst" }
            };

            var data = context.Data.LoadFromEnumerable(inputData);

            var expressions = context.Transforms.Expression("SquareRootOutput", "(x) => sqrt(x)", "YearsExperience")
                .Append(context.Transforms.Expression("TeamsManagedOutput", "(x, y) => x ? y : 0", nameof(SalaryInput.IsManager), nameof(SalaryInput.NumberOfTeamsManaged)))
                .Append(context.Transforms.Expression("ToLowerOutput", "(x) => lower(x)", nameof(SalaryInput.Title)));

            var expressionsTransformed = expressions.Fit(data).Transform(data);

            var expressionsData = context.Data.CreateEnumerable<ExpressionOutput>(expressionsTransformed,
                reuseRowObject: false);

            foreach (var expression in expressionsData)
            {
                Console.WriteLine($"Square Root - {expression.SquareRootOutput}");
                Console.WriteLine($"Teams Managed - {expression.TeamsManagedOutput}");
                Console.WriteLine($"To Lower - {expression.ToLowerOutput}");
                Console.WriteLine(Environment.NewLine);
            }

            Console.ReadLine();
        }
    }
}
