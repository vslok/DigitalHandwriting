using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace DigitalHandwriting.Services
{
    public class KeyboardMetricsCollectionService
    {
        private readonly List<List<int>> _keyPressedTimes;

        private readonly List<List<int>> _betweenKeysTimes;

        private DateTime _lastKeyDownTime;

        private int _collectingMetricsStep;

        public KeyboardMetricsCollectionService() 
        { 
            _collectingMetricsStep = 0;
            _keyPressedTimes = new List<List<int>>(3);
            _betweenKeysTimes = new List<List<int>>(3);

            for (int i = 0; i < 3; i++)
            {
                _keyPressedTimes.Add(new List<int>());
                _betweenKeysTimes.Add(new List<int>());
            }
        }

        public List<int> GetKeyPressedTimesMedians() => Calculations.CalculateMedianValue(_keyPressedTimes);
        public List<int> GetBetweenKeysTimesMedians() => Calculations.CalculateMedianValue(_betweenKeysTimes);

        public List<int> GetCurrentStepKeyPressedValues() => _keyPressedTimes[_collectingMetricsStep];
        public List<int> GetCurrentStepBetweenKeysValues() => _betweenKeysTimes[_collectingMetricsStep];

        public void OnKeyUpEvent(KeyEventArgs args)
        {
            var keyPressedTime = (DateTime.UtcNow - _lastKeyDownTime).Milliseconds;
            _keyPressedTimes[_collectingMetricsStep].Add(keyPressedTime);
        }

        public void OnKeyDownEvent(KeyEventArgs args)
        {
            if (_lastKeyDownTime.Equals(default))
            {
                _lastKeyDownTime = DateTime.UtcNow;
                return;
            }

            var time = (DateTime.UtcNow - _lastKeyDownTime).Milliseconds;
            _lastKeyDownTime = DateTime.UtcNow;
            _betweenKeysTimes[_collectingMetricsStep].Add(time);
        }

        public void IncreaseMetricsCollectingStep()
        {
            _collectingMetricsStep++;
            _lastKeyDownTime = default;
        }

        public void ResetMetricsCollection()
        {
            _collectingMetricsStep = 0;
            _lastKeyDownTime = default;
            _betweenKeysTimes.ForEach(step => step.Clear());
            _keyPressedTimes.ForEach(step => step.Clear());
        }
    }
}
