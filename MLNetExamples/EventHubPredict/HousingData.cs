using Microsoft.ML.Data;
using System.Text.Json.Serialization;

namespace EventHubPredict
{
    public class HousingData
    {
        [LoadColumn(0)]
        [JsonPropertyName("longitude")]
        public float Longitude { get; set; }

        [LoadColumn(1)]
        [JsonPropertyName("latitude")]
        public float Latitude { get; set; }

        [LoadColumn(2)]
        [JsonPropertyName("housing_median_age")]
        public float HousingMedianAge { get; set; }

        [LoadColumn(3)]
        [JsonPropertyName("total_rooms")]
        public float TotalRooms { get; set; }

        [LoadColumn(4)]
        [JsonPropertyName("total_bedrooms")]
        public float TotalBedrooms { get; set; }

        [LoadColumn(5)]
        [JsonPropertyName("population")]
        public float Population { get; set; }

        [LoadColumn(6)]
        [JsonPropertyName("households")]
        public float Households { get; set; }

        [LoadColumn(7)]
        [JsonPropertyName("median_income")]
        public float MedianIncome { get; set; }

        [LoadColumn(8), ColumnName("Label")]
        public float MedianHouseValue { get; set; }

        [LoadColumn(9)]
        [JsonPropertyName("ocean_proximity")]
        public string OceanProximity { get; set; }
    }
}
