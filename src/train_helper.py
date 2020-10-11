import torch
from util.progressbar import progress_bar
from dataset.HO_Data.convert import *
from util.contactloss import compute_contact_loss

mseLoss = torch.nn.MSELoss()
bceLoss = torch.nn.BCELoss()
bceLossNoReduce = torch.nn.BCELoss(reduction='none')
####################### Start: training and validiation loop for shapenet variants #######################

def train_HO_SNet_v1(model, criterion, optimizer, train_loader, device=torch.device('cuda')):
    model.train()
    train_loss = 0

    for batch_idx, (depth_voxel,norm_handmesh,norm_objmesh) in enumerate(train_loader):
        depth_voxel,norm_handmesh, norm_objmesh = depth_voxel.to(device),norm_handmesh.to(device),norm_objmesh.to(device)
        optimizer.zero_grad()
        out = model(depth_voxel)
        loss = criterion(out['handverts'],norm_handmesh) + criterion(out['objverts'],norm_objmesh)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))

def val_HO_SNet_v1(model, val_loader, device=torch.device('cuda')):
    model.eval()
    batch_Hmesh,batch_Omesh = 0,0

    with torch.no_grad():
        for batch_idx, (depth_voxel,GT) in enumerate(val_loader):
            depth_voxel  = depth_voxel.to(device)
            out = model(depth_voxel)
            mean_Hmesh, mean_Omesh = calculateMeshBatchLoss(out, GT)
            batch_Hmesh += mean_Hmesh
            batch_Omesh += mean_Omesh
            msg = 'Hmesh Loss: {} Omesh Loss: {}'.format(batch_Hmesh / (batch_idx + 1),batch_Omesh / (batch_idx + 1))
            progress_bar(batch_idx, len(val_loader), msg)

def train_HO_SNet_v2(model, criterion, optimizer, train_loader, device=torch.device('cuda')):
    model.train()
    train_loss = 0

    for batch_idx, (depth_voxel,heatmap_joints,heatmap_bbox,norm_handmesh,norm_objmesh) in enumerate(train_loader):

        depth_voxel, heatmap_joints,heatmap_bbox = depth_voxel.to(device), heatmap_joints.to(device), heatmap_bbox.to(device)
        norm_handmesh, norm_objmesh = norm_handmesh.to(device),norm_objmesh.to(device)
        inputs = torch.cat((depth_voxel,heatmap_joints,heatmap_bbox),dim=1)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out['handverts'],norm_handmesh) + criterion(out['objverts'],norm_objmesh)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))

def val_HO_SNet_v2(model, val_loader, device=torch.device('cuda')):
    model.eval()
    batch_Hmesh,batch_Omesh = 0,0

    with torch.no_grad():
        for batch_idx, (depth_voxel,heatmap_joints,heatmap_bbox,GT) in enumerate(val_loader):
            depth_voxel, heatmap_joints, heatmap_bbox = depth_voxel.to(device), heatmap_joints.to(device),heatmap_bbox.to(device),
            inputs = torch.cat((depth_voxel, heatmap_joints, heatmap_bbox), dim=1)
            out = model(inputs)
            mean_Hmesh, mean_Omesh = calculateMeshBatchLoss(out, GT)
            batch_Hmesh += mean_Hmesh
            batch_Omesh += mean_Omesh
            msg = 'Hmesh Loss: {} Omesh Loss: {}'.format(batch_Hmesh / (batch_idx + 1), batch_Omesh / (batch_idx + 1))
            progress_bar(batch_idx, len(val_loader), msg)

def train_HO_SNet_v3(model, criterion, optimizer, train_loader, device=torch.device('cuda')):
    model.train()
    train_loss = 0

    for batch_idx, (depth_voxel,heatmap_joints,heatmap_bbox,mesh_voxel,norm_handmesh,norm_objmesh) in enumerate(train_loader):

        depth_voxel, heatmap_joints,heatmap_bbox, mesh_voxel = depth_voxel.to(device), heatmap_joints.to(device), \
                                                               heatmap_bbox.to(device),mesh_voxel.to(device)
        norm_handmesh, norm_objmesh = norm_handmesh.to(device),norm_objmesh.to(device)
        inputs = torch.cat((depth_voxel,heatmap_joints,heatmap_bbox,mesh_voxel),dim=1)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out['handverts'],norm_handmesh) + criterion(out['objverts'],norm_objmesh)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))


def val_HO_SNet_v3(model, criterion, val_loader, device=torch.device('cuda')):
    model.eval()
    batch_Hmesh,batch_Omesh = 0,0

    with torch.no_grad():
        for batch_idx, (depth_voxel,heatmap_joints,heatmap_bbox,mesh_voxel,GT) in enumerate(val_loader):
            depth_voxel, heatmap_joints, heatmap_bbox, mesh_voxel = depth_voxel.to(device), heatmap_joints.to(device), \
                                                                    heatmap_bbox.to(device), mesh_voxel.to(device)
            norm_handmesh, norm_objmesh = norm_handmesh.to(device), norm_objmesh.to(device)
            inputs = torch.cat((depth_voxel, heatmap_joints, heatmap_bbox,mesh_voxel), dim=1)
            out = model(inputs)
            mean_Hmesh, mean_Omesh = calculateMeshBatchLoss(out, GT)
            batch_Hmesh += mean_Hmesh
            batch_Omesh += mean_Omesh
            msg = 'Hmesh Loss: {} Omesh Loss: {}'.format(batch_Hmesh / (batch_idx + 1), batch_Omesh / (batch_idx + 1))
            progress_bar(batch_idx, len(val_loader), msg)

####################### End: training and validiation loop for Shapenet variants #######################

####################### Start: training and validiation loop for Voxelnet #######################
def train_HO_VNet(model, criterion, optimizer, train_loader, device=torch.device('cuda')):
    model.train()
    train_loss = 0

    for batch_idx, (depth_voxel,heatmap_joints,heatmap_bbox,mesh_voxel) in enumerate(train_loader):

        depth_voxel, heatmap_joints,heatmap_bbox, mesh_voxel = depth_voxel.to(device), heatmap_joints.to(device), \
                                                               heatmap_bbox.to(device),mesh_voxel.to(device)
        inputs = torch.cat((depth_voxel,heatmap_joints,heatmap_bbox),dim=1)
        optimizer.zero_grad()
        out_voxel = model(inputs)
        loss = criterion(out_voxel,mesh_voxel)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))

def val_HO_VNet(model, criterion, val_loader, device=torch.device('cuda')):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (depth_voxel,heatmap_joints,heatmap_bbox,mesh_voxel) in enumerate(val_loader):
            depth_voxel, heatmap_joints, heatmap_bbox, mesh_voxel = depth_voxel.to(device), heatmap_joints.to(device), \
                                                                    heatmap_bbox.to(device), mesh_voxel.to(device)
            inputs = torch.cat((depth_voxel, heatmap_joints, heatmap_bbox), dim=1)
            out_voxel = model(inputs)
            loss = criterion(out_voxel, mesh_voxel)
            val_loss += loss.item()
            progress_bar(batch_idx, len(val_loader), 'Loss: {}'.format(val_loss/(batch_idx+1)))
            #print('loss: {}'.format(val_loss/(batch_idx+1)))
####################### End: training and validiation loop for Voxelnet #######################

####################### Start: training and validiation loop for Posenet #######################
def train_HO_PNet(model, criterion, optimizer, train_loader, device=torch.device('cuda')):
    model.train()
    train_loss = 0

    for batch_idx, (depthvoxel,heatmap_joints,heatmap_bbox) in enumerate(train_loader):
        depthvoxel, heatmap_joints, heatmap_bbox = depthvoxel.to(device), heatmap_joints.to(device), heatmap_bbox.to(device)
        optimizer.zero_grad()
        out = model(depthvoxel)
        loss = criterion(out['handpose'], heatmap_joints) + criterion(out['objpose'], heatmap_bbox)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss / (batch_idx + 1)))
        # print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))

def val_HO_PNet(model, val_loader, device=torch.device('cuda')):
    model.eval()
    batch_Hpose, batch_Opose = 0, 0

    with torch.no_grad():
        for batch_idx, (depthvoxel,GT) in enumerate(val_loader):
            depthvoxel = depthvoxel.to(device)
            out = model(depthvoxel)
            mean_Hpose,mean_Opose = calculatePoseBatchLoss(out, GT)
            batch_Hpose += mean_Hpose
            batch_Opose += mean_Opose
            msg = 'Hpose Loss: {} Opose Loss: {}'.format(batch_Hpose / (batch_idx + 1), batch_Opose / (batch_idx + 1))
            progress_bar(batch_idx, len(val_loader), msg)

####################### End: training and validiation loop for Posenet #######################

####################### start: training and validiation loop for joint networks #######################
def train_HO_study2(model, criterion1, criterion2, optimizer, train_loader, device=torch.device('cuda')):
    model.train()
    train_loss = 0

    for batch_idx, (voxel88,heatmap_joints,heatmap_bbox,voxel44,mesh_voxel,norm_handmesh,norm_objmesh) in enumerate(train_loader):

        voxel88, heatmap_joints,heatmap_bbox,voxel44 = voxel88.to(device), heatmap_joints.to(device), \
                                                               heatmap_bbox.to(device),voxel44.to(device)
        mesh_voxel,norm_handmesh, norm_objmesh = mesh_voxel.to(device),norm_handmesh.to(device),norm_objmesh.to(device)
        out = model(voxel88,voxel44,)
        poseloss = criterion1(out['handpose'],heatmap_joints) + criterion1(out['objpose'],heatmap_bbox)
        voxelloss = criterion2(out['voxel'],mesh_voxel)
        shapeloss = criterion1(out['handverts'],norm_handmesh)+criterion1(out['objverts'],norm_objmesh)
        #print('poseloss: {}, shapeloss: {}, voxelloss: {}'.format(poseloss,shapeloss,voxelloss))
        loss = poseloss + voxelloss + shapeloss
        #loss = shapeloss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))

def val_HO_study2(model, val_loader, device=torch.device('cuda')):
    model.eval()
    batch_Hpose,batch_Opose,batch_Hmesh,batch_Omesh = 0,0,0,0

    with torch.no_grad():
        for batch_idx, (voxel88,voxel44,GT) in enumerate(val_loader):
            voxel88, voxel44 = voxel88.to(device), voxel44.to(device)
            out = model(voxel88, voxel44)
            mean_Hpose,mean_Opose,mean_Hmesh,mean_Omesh = calculateBatchLoss(out,GT)
            batch_Hpose += mean_Hpose
            batch_Opose += mean_Opose
            batch_Hmesh += mean_Hmesh
            batch_Omesh += mean_Omesh
            msg = 'Hpose Loss: {} Opose Loss: {} Hmesh Loss: {} Omesh Loss: {}'.format(batch_Hpose/(batch_idx+1),batch_Opose / (batch_idx + 1),
                                                                                       batch_Hmesh / (batch_idx + 1),batch_Omesh / (batch_idx + 1))
            progress_bar(batch_idx, len(val_loader), msg)
            #print('loss: {}'.format(val_loss/(batch_idx+1)))

def train_HO_PVSNet(model, criterion1, optimizer, train_loader, device=torch.device('cuda')):
    train_loss = 0
    for batch_idx, (voxel88,voxel44,norm_handmesh,norm_objmesh) in enumerate(train_loader):

        voxel88, voxel44 = voxel88.to(device), voxel44.to(device)
        norm_handmesh, norm_objmesh = norm_handmesh.to(device),norm_objmesh.to(device)

        result = model(voxel88,voxel44)
        shapeloss = criterion1(result['handverts'],norm_handmesh)+criterion1(result['objverts'],norm_objmesh)
	############# to be added later ###############
        ######contactloss = compute_contact_loss()
        shapeloss.backward()
        optimizer.step()
        train_loss += shapeloss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))

def val_HO_PVSNet( model, val_loader, device=torch.device('cuda')):

    batch_Hmesh,batch_Omesh = 0,0

    with torch.no_grad():
        for batch_idx, (voxel88,voxel44,GT) in enumerate(val_loader):
            voxel88, voxel44 = voxel88.to(device), voxel44.to(device)
            result = model(voxel88,voxel44)
            mean_Hmesh, mean_Omesh = calculateMeshBatchLoss(result, GT)
            batch_Hmesh += mean_Hmesh
            batch_Omesh += mean_Omesh
            msg = 'Hmesh Loss: {} Omesh Loss: {}'.format(batch_Hmesh / (batch_idx + 1),batch_Omesh / (batch_idx + 1))
            progress_bar(batch_idx, len(val_loader), msg)
            #print('loss: {}'.format(val_loss/(batch_idx+1)))

def train_HO_study1(model, criterion1, optimizer1, train_loader, device=torch.device('cuda')):
    model.train()
    train_loss = 0

    for batch_idx, (voxel88,heatmap_joints,heatmap_bbox,voxel44,norm_handmesh,norm_objmesh) in enumerate(train_loader):

        voxel88, heatmap_joints,heatmap_bbox,voxel44 = voxel88.to(device), heatmap_joints.to(device), \
                                                               heatmap_bbox.to(device),voxel44.to(device)
        norm_handmesh, norm_objmesh = norm_handmesh.to(device),norm_objmesh.to(device)
        optimizer1.zero_grad()
        out = model(voxel88,voxel44)
        poseloss = criterion1(out['handpose'],heatmap_joints) + criterion1(out['objpose'],heatmap_bbox)
        shapeloss = criterion1(out['handverts'],norm_handmesh)+criterion1(out['objverts'],norm_objmesh)
        #print('poseloss: {}, shapeloss: {}'.format(poseloss, shapeloss))
        loss = shapeloss
        loss.backward()
        optimizer1.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))

def val_HO_study1(model, val_loader, device=torch.device('cuda')):
    model.eval()
    batch_Hpose,batch_Opose,batch_Hmesh,batch_Omesh = 0,0,0,0

    with torch.no_grad():
        for batch_idx, (voxel88,voxel44,GT) in enumerate(val_loader):
            voxel88, voxel44 = voxel88.to(device), voxel44.to(device)
            out = model(voxel88, voxel44)
            mean_Hpose,mean_Opose,mean_Hmesh,mean_Omesh = calculateBatchLoss(out,GT)
            batch_Hpose += mean_Hpose
            batch_Opose += mean_Opose
            batch_Hmesh += mean_Hmesh
            batch_Omesh += mean_Omesh
            msg = 'Hpose Loss: {} Opose Loss: {} Hmesh Loss: {} Omesh Loss: {}'.format(batch_Hpose/(batch_idx+1),batch_Opose / (batch_idx + 1),
                                                                                       batch_Hmesh / (batch_idx + 1),batch_Omesh / (batch_idx + 1))
            progress_bar(batch_idx, len(val_loader), msg)
            #print('loss: {}'.format(val_loss/(batch_idx+1)))

def train_HO_PVNet(model, optimizer, train_loader, device=torch.device('cuda')):
    model.train()
    train_loss = 0

    for batch_idx, (voxel88,heatmap_joints,heatmap_bbox,voxel44,mesh_voxel,pixel_weight) in enumerate(train_loader):
        voxel88, heatmap_joints, heatmap_bbox, voxel44 = voxel88.to(device), heatmap_joints.to(device), \
                                                         heatmap_bbox.to(device), voxel44.to(device)
        mesh_voxel,pixel_weight = mesh_voxel.to(device),pixel_weight.to(device)
        out = model(voxel88,voxel44,)
        poseloss = mseLoss(out['handpose'],heatmap_joints) + mseLoss(out['objpose'],heatmap_bbox)
        voxelloss = bceLossNoReduce(out['voxel'],mesh_voxel)
        #voxelloss = torch.mean(voxelloss,dim=[1,2,3,4])*pixel_weight.squeeze()
        # voxelloss = voxelloss.mean()
        voxelloss = voxelloss * pixel_weight
        voxelloss = voxelloss.mean()

        #loss_class_weighted = weighted_binary_cross_entropy(out['voxel'],mesh_voxel,pos_weight=pixel_weight)
        # loss_class_weighted.backward()
        #print('poseloss: {}, voxelloss: {}'.format(poseloss, voxelloss))
        loss = voxelloss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))

def val_HO_PVNet(model,val_loader, device=torch.device('cuda')):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (voxel88,heatmap_joints,heatmap_bbox,voxel44,mesh_voxel,class_weight) in enumerate(val_loader):
            voxel88, heatmap_joints, heatmap_bbox, voxel44 = voxel88.to(device), heatmap_joints.to(device), \
                                                             heatmap_bbox.to(device), voxel44.to(device)
            mesh_voxel = mesh_voxel.to(device)
            out = model(voxel88, voxel44, )
            poseloss = mseLoss(out['handpose'], heatmap_joints) + mseLoss(out['objpose'], heatmap_bbox)
            voxelloss = bceLoss(out['voxel'], mesh_voxel)
            loss = poseloss+voxelloss
            val_loss += loss.item()
            progress_bar(batch_idx, len(val_loader), 'Loss: {}'.format(val_loss/(batch_idx+1)))
            #print('loss: {}'.format(val_loss/(batch_idx+1)))
