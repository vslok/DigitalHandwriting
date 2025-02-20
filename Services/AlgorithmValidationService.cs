using CsvHelper;
using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using DigitalHandwriting.Repositories;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
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

    public class BiometricMetrics
    {
        public double FAR { get; set; } // False Acceptance Rate
        public double FRR { get; set; } // False Rejection Rate
        public double EER { get; set; } // Equal Error Rate
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

        public void ValidateAuthentication(string testFilePath, int n, string saveDirectory)
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

                    var user = systemUsers.Find((user) => user.Login == testAuthenticationsRecord.Login);
                    var hUserProfile = new List<List<double>>()
                    {
                        JsonSerializer.Deserialize<List<double>>(user.FirstH),
                        JsonSerializer.Deserialize<List<double>>(user.SecondH),
                        JsonSerializer.Deserialize<List<double>>(user.ThirdH),
                    };

                    var udUserProfile = new List<List<double>>()
                    {
                        JsonSerializer.Deserialize<List<double>>(user.FirstUD),
                        JsonSerializer.Deserialize<List<double>>(user.SecondUD),
                        JsonSerializer.Deserialize<List<double>>(user.ThirdUD),
                    };

                    var hUserMedian = Calculations.CalculateMedianValue(hUserProfile);
                    var udUserMedian = Calculations.CalculateMedianValue(udUserProfile);

                    var authenticationH = new List<double>(testAuthenticationsRecord.H);
                    var authenticationDU = new List<double>(testAuthenticationsRecord.UD);


                    var euclidianMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.Euclidian,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var euclidianMethodResult = euclidianMethod.Authenticate(n, authenticationH, authenticationDU);

                    var euclidianNormalizedMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.NormalizedEuclidian,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var euclidianNormalizedMethodResult = euclidianNormalizedMethod.Authenticate(n, authenticationH, authenticationDU);

                    var manhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.Manhattan,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var manhattanMethodResult = manhattanMethod.Authenticate(n, authenticationH, authenticationDU);

                    var normalizedManhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.NormalizedManhattan,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var normalizedManhattanMethodResult = normalizedManhattanMethod.Authenticate(n, authenticationH, authenticationDU);

                    var ITADMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.ITAD,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var ITADMethodResult = ITADMethod.Authenticate(n, authenticationH, authenticationDU);

                    var GunettiPicardiMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.GunettiPicardi,
                        hUserMedian,
                        udUserMedian,
                        hUserProfile,
                        udUserProfile);

                    var GunettiPicardiMethodResult = GunettiPicardiMethod.Authenticate(n, authenticationH, authenticationDU);

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
                WriteResultsToCsv(methodEntry.Value.ToList(), saveDirectory);
            }
        }

        private BiometricMetrics CalculateMetrics(List<AuthenticationValidationResult> results)
        {
            var legalUsers = results.Where(r => r.IsLegalUser).ToList();
            var impostors = results.Where(r => !r.IsLegalUser).ToList();

            // Calculate FAR (False Acceptance Rate)
            double falseAcceptances = impostors.Count(r => r.IsAuthenticated);
            double far = falseAcceptances / impostors.Count;

            // Calculate FRR (False Rejection Rate)
            double falseRejections = legalUsers.Count(r => !r.IsAuthenticated);
            double frr = falseRejections / legalUsers.Count;

            // Calculate EER (Equal Error Rate) - approximate
            double eer = (far + frr) / 2;

            return new BiometricMetrics
            {
                FAR = far * 100, // Convert to percentage
                FRR = frr * 100,
                EER = eer * 100
            };
        }

        private void WriteResultsToCsv(List<AuthenticationValidationResult> results, string saveDirectory)
        {
            if (results.Count == 0)
            {
                return;
            }

            string folderPath = Path.Combine(saveDirectory, $"N_{results[0].N}");
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }
            string filePath = Path.Combine(folderPath, $"{results[0].AuthenticationMethod}_results.csv");

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

            using (var writer = new StreamWriter(filePath))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                Console.WriteLine($"Write in file {filePath}");
                csv.WriteRecords(records);
            }
        }
    }
}
