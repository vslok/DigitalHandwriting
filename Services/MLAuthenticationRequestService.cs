using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public class PredictionResponse
    {
        public string Login { get; set; }
        [JsonPropertyName("model_type")]
        public string ModelType { get; set; }
        [JsonPropertyName("n_value")]
        public int NValue { get; set; }
        public bool Authenticated { get; set; }
    }

    public class MLAuthenticationRequestService
    {
        private readonly HttpClient _httpClient;

        public MLAuthenticationRequestService() {
            _httpClient = new HttpClient();
            _httpClient.BaseAddress = new Uri("http://localhost:5000/api/v1/");
        }

        public async Task<bool> Authenticate(
            string login,
            int nValue,
            List<double> hValues,
            List<double> udValues,
            string modelType
        ) {
            var requestData = new {
                login,
                n_value = nValue,
                h_values = hValues,
                ud_values = udValues,
                model_type = modelType
            };

            var jsonContent = new StringContent(JsonSerializer.Serialize(requestData), Encoding.UTF8, "application/json");

            try {
                var response = await _httpClient.PostAsync("predict", jsonContent);
                response.EnsureSuccessStatusCode();
                var result = await response.Content.ReadAsStringAsync();
                var predictionResponse = JsonSerializer.Deserialize<PredictionResponse>(result);
                return predictionResponse.Authenticated;
            } catch (HttpRequestException e) {
                Console.WriteLine($"Error: {e.Message}");
                return false;
            }
        }
    }
}
