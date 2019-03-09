using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace EntityFrameworkData
{
    class Program
    {
        static void Main(string[] args)
        {
            var fileData = ReadFromFile("./SalaryData.csv");


        }

        private static IEnumerable<SalaryData> ReadFromFile(string filePath)
        {
            var data = File.ReadAllLines(filePath)
                .Skip(1)
                .Select(l => l.Split(','))
                .Select(i => new SalaryData
                {
                    YearsExperience = float.Parse(i[0]),
                    Salary = float.Parse(i[1])
                });

            return data;
        }
    }
}
