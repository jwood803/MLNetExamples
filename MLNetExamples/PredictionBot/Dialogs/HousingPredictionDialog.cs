using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.Dialogs;
using Microsoft.Bot.Builder.Dialogs.Choices;
using Microsoft.Extensions.ML;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace PredictionBot.Dialogs
{
    public class HousingPredictionDialog : ComponentDialog
    {
        private readonly PredictionEnginePool<HousingData, HousingPrediction> _predictionEnginePool;

        public HousingPredictionDialog(UserState userState, PredictionEnginePool<HousingData, HousingPrediction> predictionEnginePool) 
            : base(nameof(HousingPredictionDialog))
        {
            _predictionEnginePool = predictionEnginePool;

            var steps = new WaterfallStep[]
            {
                LatitudeStepAsync,
                LongitudeStepAsync,
                HousingMedianAgeStepAsync,
                TotalRoomsStepAsync,
                TotalBedroomsStepAsync,
                PopulationStepAsync,
                HouseholdsStepAsync,
                MedianIncomeStepAsync,
                OceanProximityStepAsync,
                FinishDialogAsync
            };

            AddDialog(new WaterfallDialog(nameof(WaterfallDialog), steps));
            AddDialog(new ChoicePrompt(nameof(ChoicePrompt)));
            AddDialog(new NumberPrompt<float>(nameof(NumberPrompt<float>)));

            InitialDialogId = nameof(WaterfallDialog);
        }

        private static async Task<DialogTurnResult> LatitudeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            return await stepContext.PromptAsync(nameof(NumberPrompt<float>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the latitude?")
            });
        }

        private async Task<DialogTurnResult> LongitudeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Latitude"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<float>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the longitude?")
            });
        }

        private static async Task<DialogTurnResult> HousingMedianAgeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Longitude"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<float>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the housing median age?")
            });
        }

        private static async Task<DialogTurnResult> TotalRoomsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["MedianAge"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<float>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of total rooms?")
            });
        }

        private static async Task<DialogTurnResult> TotalBedroomsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["TotalRooms"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<float>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of total bedrooms?")
            });
        }

        private static async Task<DialogTurnResult> PopulationStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["TotalBedrooms"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<float>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the population of the area?")
            });
        }

        private static async Task<DialogTurnResult> HouseholdsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Population"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<float>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of households?")
            });
        }

        private static async Task<DialogTurnResult> MedianIncomeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Households"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<float>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the median income of the area?")
            });
        }

        private static async Task<DialogTurnResult> OceanProximityStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["MedianIncome"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(ChoicePrompt), new PromptOptions
            { 
                Prompt = MessageFactory.Text("What is the ocean proximity?"),
                Choices = ChoiceFactory.ToChoices(new List<string> { "NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND" }),
            });
        }

        private async Task<DialogTurnResult> FinishDialogAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["OceanProximity"] = stepContext.Result;

            var stepValues = stepContext.Values;

            var housingData = new HousingData
            {
                Longitude = float.Parse(stepValues["Longitude"].ToString()),
                Latitude = float.Parse(stepValues["Latitude"].ToString()),
                HousingMedianAge = float.Parse(stepValues["MedianAge"].ToString()),
                TotalRooms = float.Parse(stepValues["TotalRooms"].ToString()),
                TotalBedrooms = float.Parse(stepValues["TotalBedrooms"].ToString()),
                Population = float.Parse(stepValues["Population"].ToString()),
                Households = float.Parse(stepValues["Households"].ToString()),
                MedianIncome = float.Parse(stepValues["MedianIncome"].ToString()),
                OceanProximity = ((FoundChoice)stepValues["OceanProximity"]).Value
            };

            var prediction = _predictionEnginePool.Predict(housingData);

            await stepContext.Context.SendActivityAsync($"House value prediction is {prediction.PredictedHouseValue.ToString("C")}");

            return await stepContext.EndDialogAsync(stepContext.Values, cancellationToken: cancellationToken);
        }
    }
}
