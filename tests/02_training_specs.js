var expect = require('chai').expect
var NN = require('../index');

describe('Neural network basic training validation', function(){

	describe("Here's where I test things which are used in training, but do not test the training itself", function(){

		it("_blankify succesfully blankifies the stuff on being given values with which to update itself", function(){
			var i = new NN();
			var d = i._blankify([ [{connections:[1,2,3],weights:[1,2,3], bias:10}] ]);
			expect(d.length).to.equal(1);
			expect(d[0][0].connections.length).to.equal(3);
			expect(d[0][0].connections.length).to.equal(3);
			expect(d[0][0].connections[0]).to.equal(1)
			expect(d[0][0].connections[1]).to.equal(2)
			expect(d[0][0].connections[2]).to.equal(3)
			expect(d[0][0].weights[0]).to.equal(0)
			expect(d[0][0].weights[1]).to.equal(0)
			expect(d[0][0].weights[2]).to.equal(0)
			expect(d[0][0].bias).to.equal(0)
		});

		it("_updateWithValues succesfully adds weights / biases", function(){
			var i = new NN();
			var d = i._updateWithValues([[{connections:[1,2,3],weights:[1,2,3], bias:10}]],[[{connections:[1,2,3],weights:[1,2,3], bias:10}]]);
			expect(d.length).to.equal(1);
			expect(d[0][0].connections.length).to.equal(3);
			expect(d[0][0].connections.length).to.equal(3);
			expect(d[0][0].connections[0]).to.equal(1)
			expect(d[0][0].connections[1]).to.equal(2)
			expect(d[0][0].connections[2]).to.equal(3)
			expect(d[0][0].weights[0]).to.equal(2)
			expect(d[0][0].weights[1]).to.equal(4)
			expect(d[0][0].weights[2]).to.equal(6)
			expect(d[0][0].bias).to.equal(20)
		});

	});

	it('can train?', function(){
		var i = new NN();
		i.set('layers',[5,3,2]);
		i.init();
		i.trainBatch([[1,.5,.5,.5,1]],[[0,1]]);
	});


	var tester = function(nnet){
		for(var y = 0; y < 500; y++){
			var trainData = []
			var trainClass = []
			for(var x = 0; x < 10; x++){
				if(Math.random() > 0.5){
					trainData.push([1,0,0,0,1]);
					trainClass.push([0,1]);
				}else{
					trainData.push([0,1,1,1,0]);
					trainClass.push([1,0]);
				}
			}
			nnet.trainBatch(trainData,trainClass);
		}
		
		for(var x = 0; x < 100; x++){
			if(Math.random() > 0.5){
				var a = nnet.predict([[0,1,1,1,0]])
				if (a[0][0] < 0.9 || a[0][1] > 0.1){
					return false
				}
			}else{
				var a = nnet.predict([[1,0,0,0,1]])
				if (a[0][0] > 0.9 || a[0][1] < 0.1){
					return false
				}
			}
		}
		return true;
	}

	it('can train usefully with default settings?', function(){
		var i = new NN();
		i.set('layers',[5,5,2]);
		i.init();
		expect(tester(i)).to.eql(true);		
	});

	describe('See if it works with a bunch of different links', function(){

		it('can train usefully with tanh proportioned about 0', function(){
			var i = new NN();
			i.set('layers',[5,5,2]);
			i.set('link','tanh')
			i.set('randomness','proportionateZeroCentered');
			i.init();
			expect(tester(i)).to.eql(true);		
		});

		it('can train usefully with relu', function(){
			var i = new NN();
			i.set('layers',[5,5,2]);
			i.set('link','relu')
			i.set('randomness','proportionateZeroCentered');
			i.init();
			expect(tester(i)).to.eql(true);		
		});

		it('can train usefully with leaky relu', function(){
			var i = new NN();
			i.set('layers',[5,5,2]);
			i.set('link','leakyrelu')
			i.set('randomness','proportionatePositive');
			i.init();
			expect(tester(i)).to.eql(true);		
		});

		it('uses the cost function to give average costs over time', function(){
			var i = new NN();
			i.set('layers',[5,5,2]);
			i.set('link','leakyrelu')
			i.set('randomness','proportionatePositive');
			var trainData = []
			var trainClass = []
			for(var x = 0; x < 10; x++){
				if(Math.random() > 0.5){
					trainData.push([1,0,0,0,1]);
					trainClass.push([0,1]);
				}else{
					trainData.push([0,1,1,1,0]);
					trainClass.push([1,0]);
				}
			}
			i.init();
			var m = i.avLoss(trainData, trainClass);
			expect(tester(i)).to.eql(true);	
			var n = i.avLoss(trainData, trainClass);
			expect(tester(i)).to.eql(true);	
			var o = i.avLoss(trainData, trainClass);
			expect(m > n).to.equal(true);
			expect(n > o).to.equal(true);
		});

	});

	describe('Trying out different cost functions', function(){

		it('can train usefully with mean-hyper-cubed error', function(){
			var i = new NN();
			i.set('layers',[5,5,2]);
			i.set('link','leakyrelu')
			i.set('randomness','proportionatePositive');
			i.set('cost','mhce');
			i.init();
			expect(tester(i)).to.eql(true);		
		});

	});


	var tester2 = function(nnet){
		for(var y = 0; y < 400; y++){
			var trainData = []
			var trainClass = []
			for(var x = 0; x < 10; x++){
				if(Math.random() > 0.5){
					trainData.push([0,1,0,0,0,0,0,1,0]);
					trainClass.push([0,1]);
				}else{
					trainData.push([0,1,0,1,1,1,0,1,0]);
					trainClass.push([1,0]);
				}
			}
			nnet.trainBatch(trainData,trainClass);
		}
		
		for(var x = 0; x < 100; x++){
			if(Math.random() > 0.5){
				var a = nnet.predict([[0,1,0,0,0,0,0,1,0]])
				if (a[0][0] < 0.9 || a[0][1] > 0.1){
					return false
				}
			}else{
				var a = nnet.predict([[0,1,0,1,1,1,0,1,0]])
				if (a[0][0] > 0.9 || a[0][1] < 0.1){
					return false
				}
			}
		}
		return true;
	}

	describe('Works with some locally sparse connections', function(){

		it('can train usefully with neighbors as the "convolutor"', function(){
			var i = new NN();
			i.set('layers',[5,5,2]);
			i.set('link','leakyrelu')
			i.set('randomness','proportionatePositive');
			i.set('convolutor','neighbors');
			i.init();
			expect(tester(i)).to.eql(true);		
		});

		it('can train usefully with moore as the "convolutor"', function(){
			var i = new NN();
			i.set('layers',[5,5,2]);
			i.set('link','leakyrelu')
			i.set('randomness','proportionatePositive');
			i.set('convolutor','moore');
			i.init();
			expect(tester(i)).to.eql(true);		
		});

	});
	


});