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
        private List<List<int>> _keyPressedTimes = new List<List<int>>(3);

        private List<List<int>> _beetwenKeysTimes = new List<List<int>>(3);

        private DateTime _lastKeyDownTime;

        private int _collectingMetricsStep;

        public KeyboardMetricsCollectionService() 
        { 
            _collectingMetricsStep = 0;
        }

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
            }
            var time = (DateTime.UtcNow - _lastKeyDownTime).Milliseconds;
            _beetwenKeysTimes[_collectingMetricsStep].Add(time);
        }

        public void IncreaseMetricsCollectingStep()
        {
            _collectingMetricsStep++;
            _lastKeyDownTime = default;
        }

        public void ResetMetricsCollection()
        {
            _collectingMetricsStep = 0;
            _beetwenKeysTimes.ForEach(step => step.Clear());
            _keyPressedTimes.ForEach(step => step.Clear());
        }
    }
}
