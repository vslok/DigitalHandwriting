using DigitalHandwriting.Models;
using DigitalHandwriting.Repositories;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

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
            var systemUsers = _userRepository.getAllUsers();
            var testUsers = _dataMigrationService.GetUsersFromCsv(testFilePath);

            List<AuthenticationValidationResult> results = new List<AuthenticationValidationResult>();
            Parallel.ForEach(
                testUsers,
                (testUserRecord) =>
                {
                    var testUser = new User()
                    {
                        Login = testUserRecord.Subject,
                        Password = EncryptionService.GetPasswordHash(testUserRecord.Password, out var salt),
                        Salt = salt,
                        KeyPressedTimesMedians = JsonSerializer.Serialize(testUserRecord.ThirdH),
                        BetweenKeysTimesMedians = JsonSerializer.Serialize(testUserRecord.ThirdUD),
                        BetweenKeysPressTimesMedians = JsonSerializer.Serialize(testUserRecord.ThirdDD),
                    };

                    var user = systemUsers.Find((user) => user.Login == testUser.Login);
                    var isAuthenticated = AuthenticationService.HandwritingAuthentication(
                        user,
                        JsonSerializer.Deserialize<List<int>>(testUser.KeyPressedTimesMedians),
                        JsonSerializer.Deserialize<List<int>>(testUser.BetweenKeysTimesMedians),
                        JsonSerializer.Deserialize<List<int>>(testUser.BetweenKeysPressTimesMedians),
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
            );

            return results.OrderBy(result => result.Login).ToList();
        }
    }
}
