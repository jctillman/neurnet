var expect = require('chai').expect
var NN = require('../index');

describe('Neural network basics', function(){

	describe('Can be set up with very basics', function(){

		it('is a constructor function', function(){
			expect(typeof NN).to.equal('function');
		})

		it('can be initialized without any further work but setting layer numbers', function(){
			var a = new NN();
			expect(typeof a.init).to.equal('function');
			a.set('layers',[10,10,4]);
			a.init();
			expect(1).to.equal(1);
		});

		it('can have the link function reset to legal values, and legal values only', function(){
			var a = new NN();
			expect(a.set('link','notReal')).to.equal(false);
			expect(a.get('link')).to.not.equal('notReal');
			expect(a.set('link','sigmoid')).to.equal('sigmoid');
			expect(a.get('link')).to.equal('sigmoid');
			expect(a.set('link','tanh')).to.equal('tanh');
			expect(a.get('link')).to.equal('tanh');
			expect(a.set('link','relu')).to.equal('relu');
			expect(a.get('link')).to.equal('relu');

		});

		it('can have the randomness function set to legal values, and legal values only', function(){
			var a = new NN();
			expect(a.set('randomness','notReal')).to.equal(false);
			expect(a.get('randomness')).to.not.equal('notReal');
			expect(a.set('randomness','zero')).to.equal('zero');
			expect(a.get('randomness')).to.equal('zero');
			expect(a.set('randomness','proportionatePositive')).to.equal('proportionatePositive');
			expect(a.get('randomness')).to.equal('proportionatePositive');
			expect(a.set('randomness','proportionateZeroCentered')).to.equal('proportionateZeroCentered');
			expect(a.get('randomness')).to.equal('proportionateZeroCentered');
		});

		it('can have other important settings set to legal values, and legal values only', function(){
			var a = new NN();
			expect(a.set('layers','balderdash')).to.equal(false);
			expect(a.set('layers',[10,5,3])).to.eql([10,5,3]);
			expect(a.get('layers')).to.eql([10,5,3]);

			expect(a.set('eta','balderdash')).to.equal(false);
			expect(a.set('eta',2)).to.equal(false);
			expect(a.set('eta',0.5)).to.equal(0.5);
			expect(a.get('eta')).to.equal(0.5);
			
			

		});

		it('at least returns some *basically* valid value after being run', function(){
			var a = new NN();
			a.set('layers',[9,4,2]);
			a.init();
			var results = a.predict([ [ 1,1,1,1,1,1,1,1,1,1 ] ]);
			expect(results[0].length).to.equal(2);
			expect(results[0][0] > -1).to.equal(true);
			expect(results[0][0] < 1).to.equal(true);
			expect(results[0][1] > -1).to.equal(true);
			expect(results[0][1] < 1).to.equal(true);

			results = a.predict([ [ 0,0,0,0,0,0,0,0,0,0 ] ]);
			expect(results[0].length).to.equal(2);
			expect(results[0][0] > -1).to.equal(true);
			expect(results[0][0] < 1).to.equal(true);
			expect(results[0][1] > -1).to.equal(true);
			expect(results[0][1] < 1).to.equal(true);
		});

	});

	describe('Tests that all the "hidden" functions are returning what they should return', function(){

		it('Gives all the link functions correctly', function(){
			var a = new NN();

			//Values 
			var sig = a._getLink('sigmoid');
			var tanh = a._getLink('tanh');
			var relu = a._getLink('relu');
			expect(sig.value(0)).to.equal(0.5);
			expect(tanh.value(0)).to.equal(0);
			expect(relu.value(0)).to.equal(0);
			expect(relu.value(10)).to.equal(10);

			//Derivatives
			expect(sig.derivative(0)).to.equal(0.25);
			expect(tanh.derivative(0)).to.equal(1);
			expect(relu.derivative(-1)).to.equal(0.05);
			expect(relu.derivative(1)).to.equal(1);

		});

		it('Gives all the randomness-generating functions correctly', function(){
			var a = new NN();
			var zero = a._getRandomness('zero');
			var prop = a._getRandomness('proportionatePositive');
			var propZ = a._getRandomness('proportionateZeroCentered');
			for(var x = 0; x < 100; x++){
				expect(zero()).to.equal(0);
				expect(prop(10) < 1/Math.sqrt(10)).to.equal(true);
				expect(prop(10) > 0).to.equal(true);
				expect(propZ(10) < 1 / Math.sqrt(10)).to.equal(true);
				expect(propZ(10) > -1 / Math.sqrt(10)).to.equal(true);
			}
		});

		it('Gives all the convolution-generating functions correctly', function(){
			var a = new NN();
			var none = a._getConvolutor('none');
			var neigh = a._getConvolutor('neighbors');
			var moore = a._getConvolutor('moore');

			var x = none(0,9);
			expect(x.length).to.equal(9)
			x = none(8,9);
			expect(x.length).to.equal(9)

			var y = neigh(0,9);
			expect(y.length).to.equal(2)
			y = neigh(1,9);
			expect(y.length).to.equal(3)
			y = neigh(7,9);
			expect(y.length).to.equal(3)
			y = neigh(8,9);
			expect(y.length).to.equal(2)

			z = moore(0,9)
			expect(z.length).to.equal(4)
			z = moore(1,9)
			expect(z.length).to.equal(6)
			z = moore(2,9)
			expect(z.length).to.equal(4)

			var z = moore(4,9)
			expect(z.length).to.equal(9)

			z = moore(6,9)
			expect(z.length).to.equal(4)
			z = moore(7,9)
			expect(z.length).to.equal(6)
			z = moore(8,9)
			expect(z.length).to.equal(4)
		});

	})

});