var _ = require('lodash');

module.exports = 
		{
			eta: function(num){
				return !isNaN(num) * num <= 1;
			},
			layers: function(arr){
				return arr.constructor === Array && arr.every(function(n){return !isNaN(n) && n % 1 == 0 && n > 0;});
			},
			convolutors: {
				none: function(neuronNumber, priorLayerLength){
						return _.range(priorLayerLength);
					},
				neighbors: function(neuronNumber, priorLayerLength){
						var ret = [];
						ret.push(neuronNumber);
						(neuronNumber - 1 > -1) && ret.push(neuronNumber-1);
						(neuronNumber + 1 < priorLayerLength) && ret.push(neuronNumber+1);
						return ret;
					},
				moore: function(neuronNumber, priorLayerLength){
						//TODO: Make prettier.
						var ret = [];
						var squareSize = Math.sqrt(priorLayerLength);
						var notLeftEdge = ((neuronNumber % squareSize) !== 0);
						var notRightEdge = ((neuronNumber % squareSize !== squareSize - 1));
						var notTopEdge = ((neuronNumber + squareSize < priorLayerLength));
						var notBottomEdge = ((neuronNumber - squareSize >= 0));

						ret.push(neuronNumber)
						notLeftEdge && ret.push(neuronNumber-1);
						notRightEdge && ret.push(neuronNumber+1);
						notTopEdge && ret.push(neuronNumber+squareSize);
						notBottomEdge && ret.push(neuronNumber-squareSize);
						notLeftEdge && notTopEdge && ret.push(neuronNumber-1+squareSize);
						notRightEdge && notTopEdge && ret.push(neuronNumber+1+squareSize);
						notLeftEdge && notBottomEdge && ret.push(neuronNumber-1-squareSize);
						notRightEdge && notBottomEdge && ret.push(neuronNumber+1-squareSize);
															
						return ret;
					}
				},
			randomness: {
				zero: function (num){
					return 0;
					},
				proportionatePositive: function(num){
					return (Math.random())/Math.sqrt(num);
					},
				proportionateZeroCentered: function(num){
					return (Math.random()-0.5)/Math.sqrt(num);
					}
				},
			link: {
				sigmoid: {
					value: function(num){
						return 1 / (1 + Math.pow(Math.E, -num));
					},
					derivative: function(num){
						return 1 / (Math.pow(Math.E, -num) + 2 + Math.pow(Math.E, num))
					}
				},
				tanh: {
					value: function(num){
						var eToX = Math.pow(Math.E, num);
						var eToNegX = Math.pow(Math.E, -num);
						return (eToX - eToNegX)/(eToX + eToNegX);
					},
					derivative: function(num){
						var eTo2X = Math.pow(Math.E, 2*num);
						var eToNeg2X = Math.pow(Math.E, -2*num);
						return 4 / (eTo2X + 2 + eToNeg2X);
					}
				},
				relu: {
					value: function(num){
						return (num > 0) ? num : num / 20; 
					},
					derivative: function(num){
						return (num > 0) ? 1 : 0.05;
					}
				},
				leakyrelu: {
					value: function(num){
						return (num > 0) ? num : num / 10; 
					},
					derivative: function(num){
						return (num > 0) ? 1 : 0.1;
					}
				}
			},
			cost: {
				mse: {
					singleNeuron: function(target, actual){
						return 0.5 * Math.pow(target-actual,2);
					},
					singleDerivative: function(target){
						return function(actual){
							return - ( target - actual );
						}
					}
				},
				mhce: {
					singleNeuron: function(target, actual){
						return 0.25 * Math.pow(target-actual,4);
					},
					singleDerivative: function(target){
						return function(actual){
							return - Math.pow( target - actual, 3);
						}
					}
				}
			}
		};

