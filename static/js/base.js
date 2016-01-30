$(document).ready(function(){
    $('#predictions_table').DataTable();
    $('select').material_select();
    $('.train_loader').fadeOut();
    predictions=[];
    $('#add_predictions').click(function(){
    	$('#predictions_table').DataTable().$('input[type="checkbox"]:checked').each(function(){
    		predictions.push($(this).attr("prediction"));
    		console.log($(this).attr("prediction"));
    	});
        $('.train_loader').fadeIn();
    	$.ajax({        
            type:"POST",
            url: "/predict/add_predictions/",
            data: {
              csrfmiddlewaretoken:$("input[name=csrfmiddlewaretoken]").val(),
              'predictions': JSON.stringify(predictions),
          },

          success: function(data) {  
            console.log(data); 
            window.alert("Predictions added to training set successfully")
            $('.train_loader').fadeOut();
            window.location.reload();   
        }
    }); 
    });
})