using DigitalHandwriting.Repositories;
using System.Collections.Generic;
using System.Text.Json;

namespace DigitalHandwriting.Services
{
    public class AuthenticationValidationResult
    {
        public string Login { get; set; }
        public bool IsAuthenticated { get; set; }
        public double KeyPressedDistance { get; set; }
        public double BetweenKeysDistance { get; set; }
        public double BetweenKeysPressDistance { get; set; }
    }

    public class AlgorithmValidationService
    {
        private readonly DataMigrationService _dataMigrationService;
        private readonly UserRepository _userRepository;

        public AlgorithmValidationService()
        {
            _dataMigrationService = new DataMigrationService();
            _userRepository = new UserRepository();
        }

        public List<AuthenticationValidationResult> ValidateAuthentication(string testFilePath)
        {
            var testUsers = _dataMigrationService.ReadUsersFromCsv(testFilePath);
            List<AuthenticationValidationResult> results = new List<AuthenticationValidationResult>();
            foreach(var testUser in testUsers)
            {
                var user = _userRepository.GetUser(testUser.Login);
                var isAuthenticated = AuthenticationService.HandwritingAuthentication(
                    user,
                    JsonSerializer.Deserialize<List<int>>(testUser.KeyPressedTimes),
                    JsonSerializer.Deserialize<List<int>>(testUser.BetweenKeysTimes),
                    JsonSerializer.Deserialize<List<int>>(testUser.BetweenKeysPressTimes),
                    out var keyPressedDistance, out var betweenKeysDistance, out var betweenKeysPressDistance);

                results.Add(new AuthenticationValidationResult()
                {
                    Login = user.Login,
                    IsAuthenticated = isAuthenticated,
                    KeyPressedDistance = keyPressedDistance,
                    BetweenKeysDistance = betweenKeysDistance,
                    BetweenKeysPressDistance = betweenKeysPressDistance,
                });
            }

            return results;
        }
    }
}
