using CsvHelper; // Keep if WriteResultsToCsv uses it
using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using DigitalHandwriting.Repositories;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization; // Keep if WriteResultsToCsv uses it
using System.IO; // Keep for Path, Directory
using System.Linq; // Required for .Any()
// System.Text.Json might no longer be needed here if all direct uses are removed
using System.Threading.Tasks; // For Parallel.ForEach

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
        public Method AuthenticationMethod { get; set; }
    }

    public class BiometricMetrics // Unchanged
    {
        public double FAR { get; set; } // False Acceptance Rate
        public double FRR { get; set; } // False Rejection Rate
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

            Parallel.ForEach(
                testAuthentications,
                (testAuthenticationsRecord) =>
                {
                    void ProcessMethod(Method method, List<AuthenticationResult> currentResults, string userLogin, bool isLegalUser) // Renamed 'results' to 'currentResults'
                    {
                        foreach (var result in currentResults)
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
                    }

                    var thresholds = new List<double>() { 0.20 };
                    /*for (double t = 0.0; t <= 1.0; t += 0.01)
                    {
                        thresholds.Add(Math.Round(t, 2));
                    }*/

                    var user = systemUsers.Find((u) => u.Login == testAuthenticationsRecord.Login);

                    if (user == null)
                    {
                        Console.WriteLine($"Error: User {testAuthenticationsRecord.Login} not found in system users during validation. Skipping.");
                        return; // Skip this iteration of Parallel.ForEach
                    }

                    // Directly use HSampleValues and UDSampleValues from the User model
                    var hUserProfile = user.HSampleValues;
                    var udUserProfile = user.UDSampleValues;

                    if (hUserProfile == null || udUserProfile == null)
                    {
                        Console.WriteLine($"Error: User {user.Login} has null HSampleValues or UDSampleValues. Skipping.");
                        return; // Skip this iteration
                    }

                    // Optional: Check if lists are empty if downstream logic requires non-empty lists
                    // and if CalculateMeanValue can't handle empty List<List<double>> or inner empty List<double>
                    if (!hUserProfile.Any(list => list.Any()) || !udUserProfile.Any(list => list.Any()))
                    {
                        Console.WriteLine($"Warning: User {user.Login} has HSampleValues or UDSampleValues that are empty or contain only empty sample lists. Median calculation might be affected or fail.");
                        // Depending on Calculations.CalculateMeanValue, this might be an error.
                        // If CalculateMeanValue handles this gracefully (e.g., returns empty or NaN), this might just be a warning.
                        // If it throws, you might want to 'return;' here.
                    }

                    var hUserMedian = Calculations.CalculateMeanValue(hUserProfile); // Ensure this can handle List<List<double>>
                    var udUserMedian = Calculations.CalculateMeanValue(udUserProfile); // Ensure this can handle List<List<double>>

                    var authenticationH = new List<double>(testAuthenticationsRecord.H); // Assuming testAuthenticationsRecord.H is double[] from CSV
                    var authenticationDU = new List<double>(testAuthenticationsRecord.UD); // Assuming testAuthenticationsRecord.UD is double[]

                    // --- The factory calls remain the same as they expect List<List<double>> and List<double> ---
                    var euclidianMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.Euclidian, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
                    var euclidianMethodResult = euclidianMethod.Authenticate(n, authenticationH, authenticationDU, thresholds);

                    var euclidianNormalizedMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.NormalizedEuclidian, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
                    var euclidianNormalizedMethodResult = euclidianNormalizedMethod.Authenticate(n, authenticationH, authenticationDU, thresholds);

                    var manhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.Manhattan, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
                    var manhattanMethodResult = manhattanMethod.Authenticate(n, authenticationH, authenticationDU, thresholds);

                    var filteredManhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.FilteredManhattan, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
                    var filteredManhattanMethodResult = filteredManhattanMethod.Authenticate(n, authenticationH, authenticationDU, thresholds);

                    var scaledManhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.ScaledManhattan, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
                    var scaledManhattanMethodResult = scaledManhattanMethod.Authenticate(n, authenticationH, authenticationDU, thresholds);

                    var ITADMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                        Method.ITAD, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
                    var ITADMethodResult = ITADMethod.Authenticate(n, authenticationH, authenticationDU, thresholds);

                    ProcessMethod(Method.Euclidian, euclidianMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.NormalizedEuclidian, euclidianNormalizedMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.Manhattan, manhattanMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.FilteredManhattan, filteredManhattanMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.ScaledManhattan, scaledManhattanMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                    ProcessMethod(Method.ITAD, ITADMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
                }
            );

            foreach (var methodEntry in resultsByMethod)
            {
                WriteResultsToCsv(methodEntry.Value.ToList(), saveDirectory);
            }
        }

        private void WriteResultsToCsv(List<AuthenticationValidationResult> results, string saveDirectory) // Unchanged
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
                threshold = r.Threshold,
                r.IsAuthenticated,
                r.N,
                H_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.H),
                DU_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.UD),
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