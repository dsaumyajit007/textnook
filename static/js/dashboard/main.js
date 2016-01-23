
$(document).ready(function(){
    $('#skills').DataTable();
    $('#editSkillsTable').DataTable();
    $('#editContributorsTable').DataTable();
    $('#contributers').DataTable();
    $('select').material_select();

    skills=[];
    contributers=[];
    project={};
    $('#addProject').click(function(){
    	$('#skills').DataTable().$('input[type="checkbox"]:checked').each(function(){
    		skills.push($(this).attr("skillId"));
    		console.log($(this).attr("skillId"));
    	});
    	$('#contributers').DataTable().$('input[type="checkbox"]:checked').each(function(){
    		contributers.push($(this).attr("username"));
    	});
    	project['skills']=skills;
    	project['contributers']=contributers;
    	project['projectName']=$('#projectName').val();
    	project['projectDescription']=$('#projectDescription').val();
    	$.ajax({        
            type:"POST",
            url: "../addProject/",
            data: {
              csrfmiddlewaretoken:$("input[name=csrfmiddlewaretoken]").val(),
              'project': JSON.stringify(project),
          },
          success: function(data) {  
            console.log(data); 
            window.alert("Project added successfully")
            window.location.href = "../";    
        }
    }); 
    });

    $('.viewProject').click(function(){
    	console.log('clicked');
    	window.open("viewProject/?projectId="+$(this).attr('projectId'),"_blank");
    });
    var projectId="-1";
    $('#editProjectName').click(function(){
        $('#editProjectNameModal').openModal();
        $('#projectNameChange').val($(this).attr('projectName'));
        projectId=$(this).attr('projectId');
    });


    $('#editProjectDescription').click(function(){
        $('#editProjectDescriptionModal').openModal();
        $('#projectDescriptionChange').val($(this).attr('projectDescription'));
        projectId=$(this).attr('projectId');
    });

    $('#editSkills').click(function(){
        $('#editSkillsModal').openModal();
        // $('#projectDescriptionChange').val($(this).attr('projectDescription'));
        // projectId=$(this).attr('projectId');
    });
    $('#editContributors').click(function(){
        $('#editContributorsModal').openModal();
        // $('#projectDescriptionChange').val($(this).attr('projectDescription'));
        // projectId=$(this).attr('projectId');
    });
    $('#editProjectNameBtn').click(function(){
        $.ajax({        
            type:"POST",
            url: "../updateProject/",
            data: {
                csrfmiddlewaretoken:$("input[name=csrfmiddlewaretoken]").val(),
                'projectName': $('#projectNameChange').val(),
                'projectId' : projectId
            },
            success: function(data) {  
                console.log(data); 
                if(data == "unauthorized")
                {
                    window.alert("You are not authorized to make these changes")
                    window.open("../auth/","_blank");
                }
                else
                {
                    window.alert("Project description changed successfully");
                    window.location.reload();
                }
            }
        });  
    });    

    $('#editProjectDescriptionBtn').click(function(){
        $.ajax({        
            type:"POST",
            url: "../updateProject/",
            data: {
                csrfmiddlewaretoken:$("input[name=csrfmiddlewaretoken]").val(),
                'projectDescription': $('#projectDescriptionChange').val(),
                'projectId' : projectId
            },
            success: function(data) {  
                console.log(data); 
                if(data == "unauthorized")
                {
                    window.alert("You are not authorized to make these changes")
                    window.open("../auth/","_blank");
                }
                else
                {
                    window.alert("Project description changed successfully");
                    window.location.reload();
                }
            }
        });  
    });

    


    $("#addFiles").click(function(){
        //window.open("../testload/","_blank");
        //$('#addPicturesModel').load("../testLoad/")
        $('#addFilesModal').openModal();
        // $.ajax({        
        //         type:"GET",
        //         url: "../passProjectId/",
        //         data: {
        //                 csrfmiddlewaretoken:$("input[name=csrfmiddlewaretoken]").val(),
        //                 'projectId' : $(this).attr('projectId')
        //               },
        //         success: function(data) {  
        //             if(data=="done")
        //             {

        //                 $('#addPicturesModal').load("../uploadFiles/")
        //             }
        //         }
        //     });  

});

    $('.deleteFile').click(function(event){
        event.preventDefault();
        var input;
        var r = confirm("This file will be permanently deleted. Press cancel to keep your file");
        if (r == false) 
            return;
        
        
        var documentPath=$(this).attr('documentPath');
        var projectId=$(this).attr('projectId');
        $.ajax({
            type:"POST",
            url:"../deleteDocument/",
            data:{
             csrfmiddlewaretoken:$("input[name=csrfmiddlewaretoken]").val(),
             'documentPath':documentPath,
             'projectId':projectId
         },
         success:function(data){
            if(data=="success")
                window.alert("File deleted");
            else
                window.alert("failed to delete");
            window.location.reload();
        }
    });
    });

    $('#editSkillsBtn').click(function(){
        
        $('#editSkillsTable').DataTable().$('input[type="checkbox"]:checked').each(function(){
            skills.push($(this).attr("skillId"));
            console.log($(this).attr("skillId"));

        });
        skillList={}
        skillList['skill']=skills
        $.ajax({        
            type:"POST",
            url: "../updateProject/",
            data: {
                csrfmiddlewaretoken:$("input[name=csrfmiddlewaretoken]").val(),
                'skillList': JSON.stringify(skillList),
                'projectId' : $(this).attr('projectId')
            },
            success: function(data) {  
                console.log(data); 
                if(data == "unauthorized")
                {
                    window.alert("You are not authorized to make these changes")
                    window.open("../auth/","_blank");
                }
                else
                {
                    window.alert("Project Technologies changed successfully");
                    window.location.reload();
                }
            }
        });  
    });
    
    contributors=[]

    $('#editContributorsBtn').click(function(){
        
        $('#editContributorsTable').DataTable().$('input[type="checkbox"]:checked').each(function(){
            contributors.push($(this).attr("username"));
            console.log($(this).attr("username"));

        });
        contributorList={}
        contributorList['contributors']=contributors
        $.ajax({        
            type:"POST",
            url: "../updateProject/",
            data: {
                csrfmiddlewaretoken:$("input[name=csrfmiddlewaretoken]").val(),
                'contributorList': JSON.stringify(contributorList),
                'projectId' : $(this).attr('projectId')
            },
            success: function(data) {  
                console.log(data); 
                if(data == "unauthorized")
                {
                    window.alert("You are not authorized to make these changes")
                    window.open("../auth/","_blank");
                }
                else
                {
                    window.alert("Project Contributers changed successfully");
                    window.location.reload();
                }
            }
        });  
    });  

});
var passdata = function(safe){
    console.log(safe);
}


