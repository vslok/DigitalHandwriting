using CsvHelper;
using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Repositories;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public class AuthenticationValidationResult : AuthenticationResult
    {
        public AuthenticationValidationResult(AuthenticationResult authenticationResult, string userLogin, bool isLegalUser, Method authenticationMethod) : base(authenticationResult)
        {
            Login = userLogin;
            IsLegalUser = isLegalUser;
            AuthenticationMethod = authenticationMethod;
        }

        public string Login { get; set; }

        public bool IsLegalUser { get; set; }

        public Method AuthenticationMethod {  get; set; }
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
            var testAuthentications = _dataMigrationService.GetAuthenticationDataFromCsv(testFilePath);

            var resultsByMethod = new ConcurrentDictionary<Method, ConcurrentBag<AuthenticationValidationResult>>();
            List<AuthenticationValidationResult> results = new List<AuthenticationValidationResult>();
            Parallel.ForEach(
                testAuthentications,
                (testAuthenticationsRecord) =>
                {
                    void ProcessMethod(Method method, AuthenticationResult result, string userLogin, bool isLegalUser)
                    {
                        var validationResult = new AuthenticationValidationResult(
                            result,
                            userLogin,
                            isLegalUser,
                            method
                        );

                        resultsByMethod
                            .GetOrAdd(method, _ => new ConcurrentBag<AuthenticationValidationResult>())
                            .Add(validationResult);
                    }

                    var user = systemUsers.Find((user) => user.Login == testAuthenticationsRecord.Subject);
                    var hUserProfile = new List<List<double>>()
                    {
                        JsonSerializer.Deserialize<List<double>>(user.KeyPressedTimesFirst),
                        JsonSerializer.Deserialize<List<double>>(user.KeyPressedTimesSecond),
                        JsonSerializer.Deserialize<List<double>>(user.KeyPressedTimesThird),
                    };

                    var udUserProfile = new List<List<double>>()
                    {
                        JsonSerializer.Deserialize<List<double>>(user.BetweenKeysPressTimesFirst),
                        JsonSerializer.Deserialize<List<double>>(user.BetweenKeysPressTimesSecond),
                        JsonSerializer.Deserialize<List<double>>(user.BetweenKeysPressTimesThird),
                    };

                    var hUserMedian = Calculations.CalculateMedianValue(hUserProfile);
                    var udUserMedian = Calculations.CalculateMedianValue(udUserProfile);

                    var authenticationH = new List<double>(testAuthenticationsRecord.H);
                    var authenticationDU = new List<double>(testAuthenticationsRecord.DU);


                    var euclidianMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.Euclidian,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var euclidianMethodResult = euclidianMethod.Authenticate(1, authenticationH, authenticationDU);

                    var euclidianNormalizedMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.NormalizedEuclidian,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var euclidianNormalizedMethodResult = euclidianNormalizedMethod.Authenticate(1, authenticationH, authenticationDU);

                    var manhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.Manhattan,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var manhattanMethodResult = manhattanMethod.Authenticate(1, authenticationH, authenticationDU);

                    var normalizedManhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.NormalizedManhattan,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var normalizedManhattanMethodResult = normalizedManhattanMethod.Authenticate(1, authenticationH, authenticationDU);

                    var ITADMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.ITAD,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var ITADMethodResult = ITADMethod.Authenticate(1, authenticationH, authenticationDU);

                    var GunettiPicardiMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.GunettiPicardi,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var GunettiPicardiMethodResult = GunettiPicardiMethod.Authenticate(1, authenticationH, authenticationDU);

                    ProcessMethod(Method.Euclidian, euclidianMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.NormalizedEuclidian, euclidianNormalizedMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.Manhattan, manhattanMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.NormalizedManhattan, normalizedManhattanMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.ITAD, ITADMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.GunettiPicardi, GunettiPicardiMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                }
            );

            foreach (var methodEntry in resultsByMethod)
            {
                WriteResultsToCsv(methodEntry.Value.ToList(), methodEntry.Key.ToString());
            }

            return resultsByMethod
                .SelectMany(x => x.Value)
                .OrderBy(result => result.AuthenticationMethod)
                .ToList();
        }

        private void WriteResultsToCsv(List<AuthenticationValidationResult> results, string methodName)
        {
            var records = results.Select(r => new
            {
                r.Login,
                r.IsLegalUser,
                AuthenticationMethod = r.AuthenticationMethod.ToString(),
                r.IsAuthenticated,
                r.N,
                H_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.H),
                DU_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.DU),
                UU_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.UU),
                DD_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.DD),
                r.TotalAuthenticationScore
            });

            string filePath = $"{methodName}_results.csv";

            using (var writer = new StreamWriter(filePath))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(records);
            }
        }
    }
}
