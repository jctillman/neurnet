var _ = require('lodash');
var statik = require('./statik');

(function(){

	function NeurNet(){

		var settings = {};
		var memory = {};

		var InnerCreate = function(){
			//enum-like values
			settings.convolutor = 'none';
			settings.link = 'sigmoid';
			settings.randomness = 'proportionatePositive';
			settings.cost = 'mse';
			//enum-unlike values 
			settings.layers = [10,5,3];
			settings.eta = 0.1;
		}

		var handle = new InnerCreate();

		handle._getLink = function(link){ return statik.link.hasOwnProperty(link) && statik.link[link]}
		handle._getConvolutor = function(convolutor){ return statik.convolutors.hasOwnProperty(convolutor) && statik.convolutors[convolutor]}
		handle._getRandomness = function(randomness){ return statik.randomness.hasOwnProperty(randomness) && statik.randomness[randomness]; }
		handle._getCost = function(cost){ return statik.cost.hasOwnProperty(cost) && statik.cost[cost];}

		handle.set = function(property, name){
			return statik.hasOwnProperty(property) && ( ( statik[property].hasOwnProperty(name) ) || (typeof statik[property] === 'function' && statik[property](name) == true)) ? settings[property] = name : false;
		}

		handle.get = function(property){
			return ( statik.hasOwnProperty(property) ) ? settings[property] : false;
		}

		handle.init = function(){
			var self = this;
			var randomness = self._getRandomness(settings.randomness);
			var convolutor = self._getConvolutor(settings.convolutor);
			memory.neurons = settings.layers.slice(1,settings.layers.length).map(function(totalNeuronNumber, layerNumber){
				return _.range(totalNeuronNumber).map(function(thisNeuronNumber){
					var connectionsToPriorLayer = (layerNumber == 0)
								? convolutor(thisNeuronNumber, settings.layers[layerNumber])
								: _.range(settings.layers[layerNumber]);
					return {
						connections: connectionsToPriorLayer,
						weights: _.range(connectionsToPriorLayer.length).map(function(){ return randomness(connectionsToPriorLayer.length)}),
						bias: randomness(1)
					}
				});
			});
		}

		handle._feedForward = function(inputArr){
			var self = this;
			var link = self._getLink(settings.link).value;
			return memory.neurons.reduce(function(allPriorActivations, thisLayerNeurons, priorlayerIndex){
				return allPriorActivations.concat([thisLayerNeurons.map(function(singleNeuron){
					var z = singleNeuron.weights.reduce(function(sum, weight, index){ return sum+weight*allPriorActivations[priorlayerIndex][singleNeuron.connections[index]].a;});
					return {
						z: z,
						a: link(z),
						weights: _.clone(singleNeuron.weights),
						connections: _.clone(singleNeuron.connections),
						bias: singleNeuron.bias
					}
				})]);
			}, [inputArr.map(function(n){return { a: n }})]);
		}

		handle._predictSingle = function(inputArr){
			var self = this;
			return self._feedForward(inputArr).slice(-1)[0].map(function(n){return n.a});
		}

		handle.predict = function(inputArrs){
			var self = this;
			return inputArrs.map(function(inputArr){return self._predictSingle(inputArr); });
		}

		handle._blankify = function(allNeurons){
			//console.log(allNeurons);
			return allNeurons.map(function(neuronLayer){
				return neuronLayer.map(function(neuron){
					return {
						connections: neuron.connections,
						weights: neuron.weights.map(function(){return 0;}),
						bias: 0
					}
				});
			});
		}

		handle._updateWithValues = function(oldNeurons, updateValues){
			var on = _.clone(oldNeurons, true);
			updateValues.forEach(function(deltaLayer, deltaIndex){
				deltaLayer.forEach(function(deltaNeuron, neuronIndex){
					on[deltaIndex][neuronIndex].weights = on[deltaIndex][neuronIndex].weights.map(function(weight, weightIndex){ return weight + deltaNeuron.weights[weightIndex] });
					on[deltaIndex][neuronIndex].bias = on[deltaIndex][neuronIndex].bias + deltaNeuron.bias;
				});
			});
			return on;
		}
		
		handle._findGradient = function(fedForward, outp, linkDerivative, costDerivative){
			return fedForward.reverse().map(function(neuronLayer, layerIndex, allFeedForward){
						return (layerIndex == 0)
								? neuronLayer.map(function(neuron, neuronIndex){
										neuron.d = linkDerivative(neuron.z)*costDerivative(outp[neuronIndex])(neuron.a);
										return neuron
									})
								: neuronLayer.map(function(neuron, neuronIndex){
									neuron.d = linkDerivative(neuron.z) * allFeedForward[layerIndex-1].reduce(function(sum, laterNeuron, laterNeuronIndex){
										return (laterNeuron.connections.indexOf(neuronIndex) !== -1) ? sum + laterNeuron.d * laterNeuron.weights[laterNeuron.connections.indexOf(neuronIndex)] : sum;
									}, 0 );
									return neuron;
								})
					});
		};

		handle._findDeltas = function(fedForward, eta){
			return fedForward.map(function(neuronLayer, layerIndex, allFeedForward){
						return (layerIndex < allFeedForward.length - 1) ? neuronLayer.map(function(neuron, neuronIndex){
							neuron.bias = -neuron.bias * eta;
							neuron.weights = neuron.weights.map(function(weight, weightIndex){ return -allFeedForward[layerIndex+1][neuron.connections[weightIndex]].a * neuron.d * eta });
							return neuron;
						}) : neuronLayer;
					}).reverse();
		}

		handle.trainBatch = function(inputArrs, outputArrs){
			var self = this;
			var linkDerivative = self._getLink(settings.link).derivative;
			var costDerivative = self._getCost(settings.cost).singleDerivative;
			memory.neurons = self._updateWithValues(memory.neurons, inputArrs.reduce(function(delta, inputArr, caseIndex){
					var inpu = inputArr;
					var outp = outputArrs[caseIndex];
					var gradients = self._findGradient(self._feedForward(inpu), outp, linkDerivative, costDerivative);	
					var deltas = self._findDeltas(gradients, settings.eta)
					return self._updateWithValues(delta, deltas.slice(1,deltas.length));			
				}, self._blankify(memory.neurons)));
		}


		handle.avLoss = function(inputArrs, idealOutputs){
			var self = this;
			var actualOutputs = self.predict(inputArrs);
			var cost = self._getCost(settings.cost).singleNeuron;
			return actualOutputs.reduce(function(outerSum, output, outerIndex){
				return outerSum + output.reduce(function(innerSum, neuron, innerIndex){
					return innerSum + cost(idealOutputs[outerIndex][innerIndex],neuron);
				}, 0) / output.length;
			}, 0) / actualOutputs.length;
		}

		handle.exportToString = function(){
			return JSON.stringify({memory: memory, settings: settings});
		}

		handle.inportFromString = function(str){
			var obj = JSON.parse(str);
			settings = obj.settings;
			memory = obj.memory;
		}
		

		return handle;
	}

	module.exports = NeurNet;

})();




