var expect = require('chai').expect
var NN = require('../index');

describe('Neural network works with more advanced settings', function(){

	describe('Can save and load stuff from strings', function(){

		it('Can make, and load from, stuff', function(){

			var a = new NN();
			a.set('layers',[9,4,2]);
			a.set('cost','mhce');
			a.set('link','leakyrelu')
			a.init();
			var str = a.exportToString();
			
			var b = new NN();
			b.init();
			b.inportFromString(str);
			expect(b.get('cost')).to.equal('mhce');
			expect(b.get('link')).to.equal('leakyrelu');

		});

		

	});

});